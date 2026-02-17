/// QOI - Quite OK Image Format
/// Encoder and decoder following the QOI specification (https://qoiformat.org).
const std = @import("std");

pub const Header = struct {
    width: u32,
    height: u32,
    channels: u8,
    colorspace: u8,

    const magic = "qoif";
    const size = 14;

    pub fn read(data: []const u8) !Header {
        if (data.len < size) return error.InvalidData;
        if (!std.mem.eql(u8, data[0..4], magic)) return error.InvalidData;
        return .{
            .width = std.mem.readInt(u32, data[4..8], .big),
            .height = std.mem.readInt(u32, data[8..12], .big),
            .channels = data[12],
            .colorspace = data[13],
        };
    }

    pub fn write(self: Header, out: []u8) void {
        @memcpy(out[0..4], magic);
        std.mem.writeInt(u32, out[4..8], self.width, .big);
        std.mem.writeInt(u32, out[8..12], self.height, .big);
        out[12] = self.channels;
        out[13] = self.colorspace;
    }
};

const Pixel = struct {
    r: u8 = 0,
    g: u8 = 0,
    b: u8 = 0,
    a: u8 = 255,

    fn hash(self: Pixel) u6 {
        return @truncate(@as(u32, self.r) *% 3 +% @as(u32, self.g) *% 5 +% @as(u32, self.b) *% 7 +% @as(u32, self.a) *% 11);
    }

    fn eql(a: Pixel, b: Pixel) bool {
        return a.r == b.r and a.g == b.g and a.b == b.b and a.a == b.a;
    }
};

const op_index: u8 = 0x00; // 2-bit tag: 00
const op_diff: u8 = 0x40; // 2-bit tag: 01
const op_luma: u8 = 0x80; // 2-bit tag: 10
const op_run: u8 = 0xc0; // 2-bit tag: 11
const op_rgb: u8 = 0xfe;
const op_rgba: u8 = 0xff;
const mask_2: u8 = 0xc0;

const end_marker = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 1 };

pub const Image = struct {
    width: u32,
    height: u32,
    channels: u8,
    pixels: []u8,

    pub fn deinit(self: *Image, allocator: std.mem.Allocator) void {
        allocator.free(self.pixels);
        self.* = undefined;
    }
};

/// Decode QOI image from bytes in memory.
pub fn decode(allocator: std.mem.Allocator, data: []const u8) !Image {
    const header = try Header.read(data);
    if (header.channels != 3 and header.channels != 4) return error.InvalidData;
    if (header.width == 0 or header.height == 0) return error.InvalidData;

    const total_pixels: usize = @as(usize, header.width) * @as(usize, header.height);
    const pixel_bytes = total_pixels * header.channels;
    const pixels = try allocator.alloc(u8, pixel_bytes);
    errdefer allocator.free(pixels);

    var index: [64]Pixel = @splat(Pixel{});
    var prev: Pixel = .{};
    var pos: usize = Header.size;
    var px_idx: usize = 0;

    while (px_idx < total_pixels) {
        if (pos >= data.len - 8) return error.InvalidData;
        const b1 = data[pos];

        if (b1 == op_rgb) {
            prev.r = data[pos + 1];
            prev.g = data[pos + 2];
            prev.b = data[pos + 3];
            pos += 4;
        } else if (b1 == op_rgba) {
            prev.r = data[pos + 1];
            prev.g = data[pos + 2];
            prev.b = data[pos + 3];
            prev.a = data[pos + 4];
            pos += 5;
        } else if (b1 & mask_2 == op_index) {
            prev = index[@as(u6, @truncate(b1))];
            pos += 1;
        } else if (b1 & mask_2 == op_diff) {
            prev.r +%= ((b1 >> 4) & 0x03) -% 2;
            prev.g +%= ((b1 >> 2) & 0x03) -% 2;
            prev.b +%= (b1 & 0x03) -% 2;
            pos += 1;
        } else if (b1 & mask_2 == op_luma) {
            const b2 = data[pos + 1];
            const dg: u8 = (b1 & 0x3f) -% 32;
            prev.r +%= dg -% 8 +% ((b2 >> 4) & 0x0f);
            prev.g +%= dg;
            prev.b +%= dg -% 8 +% (b2 & 0x0f);
            pos += 2;
        } else if (b1 & mask_2 == op_run) {
            var run: usize = @as(usize, b1 & 0x3f) + 1;
            pos += 1;
            while (run > 0) : (run -= 1) {
                const base = px_idx * header.channels;
                pixels[base] = prev.r;
                pixels[base + 1] = prev.g;
                pixels[base + 2] = prev.b;
                if (header.channels == 4) pixels[base + 3] = prev.a;
                px_idx += 1;
            }
            index[prev.hash()] = prev;
            continue;
        }

        index[prev.hash()] = prev;
        const base = px_idx * header.channels;
        pixels[base] = prev.r;
        pixels[base + 1] = prev.g;
        pixels[base + 2] = prev.b;
        if (header.channels == 4) pixels[base + 3] = prev.a;
        px_idx += 1;
    }

    return .{
        .width = header.width,
        .height = header.height,
        .channels = header.channels,
        .pixels = pixels,
    };
}

/// Encode pixel data to QOI format.
/// `pixels` must be packed RGB (channels=3) or RGBA (channels=4) bytes.
pub fn encode(allocator: std.mem.Allocator, pixels: []const u8, width: u32, height: u32, channels: u8) ![]u8 {
    if (channels != 3 and channels != 4) return error.InvalidData;
    const total_pixels: usize = @as(usize, width) * @as(usize, height);
    if (pixels.len != total_pixels * channels) return error.InvalidData;

    // Worst case: header + (1 + channels) per pixel + end marker
    const max_size = Header.size + total_pixels * (1 + @as(usize, channels)) + end_marker.len;
    var out = try allocator.alloc(u8, max_size);
    errdefer allocator.free(out);

    const header: Header = .{
        .width = width,
        .height = height,
        .channels = channels,
        .colorspace = 0,
    };
    header.write(out[0..Header.size]);

    var index: [64]Pixel = @splat(Pixel{});
    var prev: Pixel = .{};
    var run: u8 = 0;
    var pos: usize = Header.size;

    for (0..total_pixels) |px_idx| {
        const base = px_idx * channels;
        const cur: Pixel = .{
            .r = pixels[base],
            .g = pixels[base + 1],
            .b = pixels[base + 2],
            .a = if (channels == 4) pixels[base + 3] else 255,
        };

        if (cur.eql(prev)) {
            run += 1;
            if (run == 62) {
                out[pos] = op_run | (run - 1);
                pos += 1;
                run = 0;
            }
            continue;
        }

        if (run > 0) {
            out[pos] = op_run | (run - 1);
            pos += 1;
            run = 0;
        }

        const idx = cur.hash();
        if (index[idx].eql(cur)) {
            out[pos] = op_index | idx;
            pos += 1;
        } else {
            index[idx] = cur;

            if (cur.a == prev.a) {
                const dr: i8 = @as(i8, @bitCast(cur.r -% prev.r));
                const dg: i8 = @as(i8, @bitCast(cur.g -% prev.g));
                const db: i8 = @as(i8, @bitCast(cur.b -% prev.b));

                if (dr >= -2 and dr <= 1 and dg >= -2 and dg <= 1 and db >= -2 and db <= 1) {
                    out[pos] = op_diff |
                        @as(u8, @bitCast(@as(i8, dr) +% 2)) << 4 |
                        @as(u8, @bitCast(@as(i8, dg) +% 2)) << 2 |
                        @as(u8, @bitCast(@as(i8, db) +% 2));
                    pos += 1;
                } else {
                    const dr_dg: i8 = dr -% dg;
                    const db_dg: i8 = db -% dg;
                    if (dg >= -32 and dg <= 31 and dr_dg >= -8 and dr_dg <= 7 and db_dg >= -8 and db_dg <= 7) {
                        out[pos] = op_luma | @as(u8, @bitCast(@as(i8, dg) +% 32));
                        out[pos + 1] = @as(u8, @bitCast(@as(i8, dr_dg) +% 8)) << 4 |
                            @as(u8, @bitCast(@as(i8, db_dg) +% 8));
                        pos += 2;
                    } else {
                        out[pos] = op_rgb;
                        out[pos + 1] = cur.r;
                        out[pos + 2] = cur.g;
                        out[pos + 3] = cur.b;
                        pos += 4;
                    }
                }
            } else {
                out[pos] = op_rgba;
                out[pos + 1] = cur.r;
                out[pos + 2] = cur.g;
                out[pos + 3] = cur.b;
                out[pos + 4] = cur.a;
                pos += 5;
            }
        }
        prev = cur;
    }

    if (run > 0) {
        out[pos] = op_run | (run - 1);
        pos += 1;
    }

    @memcpy(out[pos..][0..end_marker.len], &end_marker);
    pos += end_marker.len;

    // Shrink to actual size
    if (allocator.resize(out, pos)) {
        return out[0..pos];
    } else {
        const trimmed = try allocator.alloc(u8, pos);
        @memcpy(trimmed, out[0..pos]);
        allocator.free(out);
        return trimmed;
    }
}
