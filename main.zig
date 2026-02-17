const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;
const qoi = @import("qoi.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

// --- Tensor helpers ---

/// Reflection padding for 2D spatial dims.
/// Input: [B, C, H, W], pads spatial dims (H and W) by `p` on each side.
fn reflectionPad2d(x: zml.Tensor, comptime p: comptime_int) zml.Tensor {
    if (p == 0) return x;

    // Pad H (axis 2): reflect top p rows and bottom p rows
    var result = reflectAxis(x, 2, p);
    // Pad W (axis 3): reflect left p cols and right p cols
    result = reflectAxis(result, 3, p);
    return result;
}

fn reflectAxis(x: zml.Tensor, comptime axis: comptime_int, comptime p: comptime_int) zml.Tensor {
    // For reflection padding of size p along `axis`:
    // Take the first p+1..1 elements (reversed) and last -1..-p-1 elements (reversed)
    // Concatenate: [reflected_low, original, reflected_high]
    const low = x.slice1d(axis, .{ .start = 1, .end = p + 1 }).reverse(.{axis});
    const high = x.slice1d(axis, .{ .start = -(p + 1), .end = -1 }).reverse(.{axis});
    return zml.Tensor.concatenate(&.{ low, x, high }, axis);
}

/// Instance normalization (no learned parameters).
/// Input: [B, C, H, W]. Normalizes over H and W per channel per batch element.
fn instanceNorm2d(x: zml.Tensor) zml.Tensor {
    const eps: f32 = 1e-5;
    // Mean over spatial dims (H=axis 2, W=axis 3)
    const mean_h = x.mean(2); // [B, C, 1, W]
    const mean_hw = mean_h.mean(3); // [B, C, 1, 1]
    const centered = x.sub(mean_hw.broad(x.shape()));
    const variance = centered.mul(centered).mean(2).mean(3); // [B, C, 1, 1]
    const inv_std = zml.Tensor.rsqrt(variance.addConstant(eps));
    return centered.mul(inv_std.broad(x.shape()));
}

/// Conv2d with bias. Expects NCHW layout.
fn conv2d(input: zml.Tensor, weight: zml.Tensor, bias: zml.Tensor, opts: struct {
    stride: i64 = 1,
    padding: i64 = 0,
}) zml.Tensor {
    const p = opts.padding;
    var result = input.conv2d(weight, .{
        .window_strides = &.{ opts.stride, opts.stride },
        .padding = &.{ p, p, p, p },
    });
    // bias shape: [C_out] -> broadcast to [B, C_out, H_out, W_out]
    result = result.add(bias.broadcast(result.shape(), &.{1}));
    return result;
}

/// Transposed Conv2d with bias. Implements ConvTranspose2d(in, out, 3, stride=2, padding=1, output_padding=1).
/// In StableHLO, transposed conv = conv with lhs_dilation (dilated input).
fn convTranspose2d(input: zml.Tensor, weight: zml.Tensor, bias: zml.Tensor) zml.Tensor {
    // ConvTranspose2d(in, out, kernel=3, stride=2, padding=1, output_padding=1)
    // In StableHLO terms:
    //   lhs_dilation = stride = 2
    //   padding_low = kernel - 1 - padding = 3 - 1 - 1 = 1
    //   padding_high = kernel - 1 - padding + output_padding = 3 - 1 - 1 + 1 = 2
    //   kernel dims swapped (input_feature=0, output_feature=1) for transposed conv
    //   window_reversal=true to flip the kernel
    var result = input.conv2d(weight, .{
        .lhs_dilation = &.{ 2, 2 },
        .padding = &.{ 1, 2, 1, 2 },
        .window_reversal = &.{ true, true },
        .kernel_input_feature_dimension = 0,
        .kernel_output_feature_dimension = 1,
    });
    result = result.add(bias.broadcast(result.shape(), &.{1}));
    return result;
}

// --- Model definition ---

const ResidualBlock = struct {
    conv1_weight: zml.Tensor,
    conv1_bias: zml.Tensor,
    conv2_weight: zml.Tensor,
    conv2_bias: zml.Tensor,

    pub fn init(view: zml.io.TensorStore.View) ResidualBlock {
        return .{
            .conv1_weight = view.createTensor("conv_block.1.weight"),
            .conv1_bias = view.createTensor("conv_block.1.bias"),
            .conv2_weight = view.createTensor("conv_block.5.weight"),
            .conv2_bias = view.createTensor("conv_block.5.bias"),
        };
    }

    pub fn forward(self: ResidualBlock, x: zml.Tensor) zml.Tensor {
        var out = reflectionPad2d(x, 1);
        out = conv2d(out, self.conv1_weight, self.conv1_bias, .{});
        out = instanceNorm2d(out);
        out = out.relu();
        out = reflectionPad2d(out, 1);
        out = conv2d(out, self.conv2_weight, self.conv2_bias, .{});
        out = instanceNorm2d(out);
        return x.add(out);
    }
};

const Generator = struct {
    // model0: ReflectionPad2d(3) -> Conv2d(3, 64, 7) -> InstanceNorm2d(64) -> ReLU
    model0_conv_weight: zml.Tensor,
    model0_conv_bias: zml.Tensor,

    // model1: Downsampling
    // Conv2d(64, 128, 3, stride=2, padding=1) -> InstanceNorm2d(128) -> ReLU
    model1_conv0_weight: zml.Tensor,
    model1_conv0_bias: zml.Tensor,
    // Conv2d(128, 256, 3, stride=2, padding=1) -> InstanceNorm2d(256) -> ReLU
    model1_conv1_weight: zml.Tensor,
    model1_conv1_bias: zml.Tensor,

    // model2: 3x ResidualBlock(256)
    res_block0: ResidualBlock,
    res_block1: ResidualBlock,
    res_block2: ResidualBlock,

    // model3: Upsampling
    // ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1) -> InstanceNorm2d(128) -> ReLU
    model3_conv0_weight: zml.Tensor,
    model3_conv0_bias: zml.Tensor,
    // ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1) -> InstanceNorm2d(64) -> ReLU
    model3_conv1_weight: zml.Tensor,
    model3_conv1_bias: zml.Tensor,

    // model4: ReflectionPad2d(3) -> Conv2d(64, 1, 7) -> Sigmoid
    model4_conv_weight: zml.Tensor,
    model4_conv_bias: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) Generator {
        return .{
            .model0_conv_weight = store.createTensor("model0.1.weight"),
            .model0_conv_bias = store.createTensor("model0.1.bias"),
            .model1_conv0_weight = store.createTensor("model1.0.weight"),
            .model1_conv0_bias = store.createTensor("model1.0.bias"),
            .model1_conv1_weight = store.createTensor("model1.3.weight"),
            .model1_conv1_bias = store.createTensor("model1.3.bias"),
            .res_block0 = ResidualBlock.init(store.withPrefix("model2.0")),
            .res_block1 = ResidualBlock.init(store.withPrefix("model2.1")),
            .res_block2 = ResidualBlock.init(store.withPrefix("model2.2")),
            .model3_conv0_weight = store.createTensor("model3.0.weight"),
            .model3_conv0_bias = store.createTensor("model3.0.bias"),
            .model3_conv1_weight = store.createTensor("model3.3.weight"),
            .model3_conv1_bias = store.createTensor("model3.3.bias"),
            .model4_conv_weight = store.createTensor("model4.1.weight"),
            .model4_conv_bias = store.createTensor("model4.1.bias"),
        };
    }

    pub fn forward(self: Generator, input: zml.Tensor) zml.Tensor {
        // model0: ReflectionPad2d(3) -> Conv2d(3, 64, 7) -> InstanceNorm2d -> ReLU
        var x = reflectionPad2d(input, 3);
        x = conv2d(x, self.model0_conv_weight, self.model0_conv_bias, .{});
        x = instanceNorm2d(x);
        x = x.relu();

        // model1: Downsampling (2 layers, stride 2, padding 1)
        x = conv2d(x, self.model1_conv0_weight, self.model1_conv0_bias, .{ .stride = 2, .padding = 1 });
        x = instanceNorm2d(x);
        x = x.relu();
        x = conv2d(x, self.model1_conv1_weight, self.model1_conv1_bias, .{ .stride = 2, .padding = 1 });
        x = instanceNorm2d(x);
        x = x.relu();

        // model2: 3 residual blocks
        x = self.res_block0.forward(x);
        x = self.res_block1.forward(x);
        x = self.res_block2.forward(x);

        // model3: Upsampling (2 transposed conv layers)
        x = convTranspose2d(x, self.model3_conv0_weight, self.model3_conv0_bias);
        x = instanceNorm2d(x);
        x = x.relu();
        x = convTranspose2d(x, self.model3_conv1_weight, self.model3_conv1_bias);
        x = instanceNorm2d(x);
        x = x.relu();

        // model4: ReflectionPad2d(3) -> Conv2d(64, 1, 7) -> Sigmoid
        x = reflectionPad2d(x, 3);
        x = conv2d(x, self.model4_conv_weight, self.model4_conv_bias, .{});
        return x.sigmoid();
    }

    pub fn load(
        self: *const Generator,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Generator) {
        return zml.io.load(Generator, self, allocator, io, platform, .{
            .store = store,
            .parallelism = 4,
            .dma_chunks = 8,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Generator)) void {
        inline for (std.meta.fields(zml.Bufferized(Generator))) |field| {
            const val = &@field(self, field.name);
            if (field.type == zml.Buffer) {
                val.deinit();
            } else if (@typeInfo(field.type) == .@"struct") {
                inline for (std.meta.fields(field.type)) |inner_field| {
                    if (inner_field.type == zml.Buffer) {
                        @field(val, inner_field.name).deinit();
                    }
                }
            }
        }
    }
};

// --- Image I/O ---

/// Load a QOI image and convert to [1, 3, H, W] f32 tensor data (values in [0, 1]).
fn loadInputImage(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !struct {
    data: []f32,
    width: u32,
    height: u32,
} {
    // Read entire file into memory
    const file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
    defer file.close(io);
    const stat = try file.stat(io);
    const file_data = try allocator.alloc(u8, stat.size);
    defer allocator.free(file_data);
    _ = try file.readPositionalAll(io, file_data, 0);

    // Decode QOI
    var image = try qoi.decode(allocator, file_data);
    defer image.deinit(allocator);

    const w = image.width;
    const h = image.height;
    const channels: u32 = image.channels;

    // Convert to f32 [1, 3, H, W] (CHW layout, normalized to [0, 1])
    const pixel_count: usize = @as(usize, w) * @as(usize, h);
    const tensor_data = try allocator.alloc(f32, 3 * pixel_count);

    for (0..h) |y| {
        for (0..w) |x| {
            const src_idx = (y * w + x) * channels;
            const dst_idx = y * w + x;
            // R channel
            tensor_data[0 * pixel_count + dst_idx] = @as(f32, @floatFromInt(image.pixels[src_idx])) / 255.0;
            // G channel
            tensor_data[1 * pixel_count + dst_idx] = @as(f32, @floatFromInt(image.pixels[src_idx + 1])) / 255.0;
            // B channel
            tensor_data[2 * pixel_count + dst_idx] = @as(f32, @floatFromInt(image.pixels[src_idx + 2])) / 255.0;
        }
    }

    return .{ .data = tensor_data, .width = w, .height = h };
}

/// Save a [1, 1, H, W] f32 tensor as a QOI grayscale image (written as RGB with R=G=B).
/// `tensor_w`/`tensor_h` are the actual tensor dimensions; `out_w`/`out_h` are the
/// desired output dimensions (crop to original input size when they differ).
fn saveOutputImage(allocator: std.mem.Allocator, io: std.Io, path: []const u8, data: []const f32, tensor_w: u32, tensor_h: u32, out_w: u32, out_h: u32) !void {
    const w = @min(tensor_w, out_w);
    const h = @min(tensor_h, out_h);
    const pixel_count: usize = @as(usize, w) * @as(usize, h);
    const rgb_pixels = try allocator.alloc(u8, pixel_count * 3);
    defer allocator.free(rgb_pixels);

    for (0..h) |y| {
        for (0..w) |x| {
            const src_idx = y * tensor_w + x;
            const dst_idx = y * w + x;
            const val: u8 = @intFromFloat(std.math.clamp(data[src_idx] * 255.0, 0.0, 255.0));
            rgb_pixels[dst_idx * 3 + 0] = val;
            rgb_pixels[dst_idx * 3 + 1] = val;
            rgb_pixels[dst_idx * 3 + 2] = val;
        }
    }

    const encoded = try qoi.encode(allocator, rgb_pixels, w, h, 3);
    defer allocator.free(encoded);

    const file = try std.Io.Dir.createFile(.cwd(), io, path, .{});
    defer file.close(io);
    try file.writePositionalAll(io, encoded, 0);
}

// --- Main ---

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    // Parse args
    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len != 4) {
        log.err("Usage: informativedrawings <model.safetensors> <input.qoi> <output.qoi>", .{});
        return error.InvalidArguments;
    }
    const model_path = process_args[1];
    const input_path = process_args[2];
    const output_path = process_args[3];

    // Load model registry
    log.info("Loading model from {s}...", .{model_path});
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, model_path);
    defer registry.deinit();

    // Init model
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    const model: Generator = .init(store.view());

    // Load input image to determine dimensions
    log.info("Loading input image {s}...", .{input_path});
    const img = try loadInputImage(allocator, io, input_path);
    defer allocator.free(img.data);
    log.info("Input image: {}x{}", .{ img.width, img.height });

    // Auto-select platform
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    // Compile model with the actual input dimensions
    const input_shape: zml.Tensor = .init(.{ 1, 3, img.height, img.width }, .f32);
    var exe = blk: {
        log.info("Compiling model...", .{});
        const start: std.Io.Timestamp = .now(io, .awake);
        defer log.info("Compiled model [{D}]", .{stdx.fmt.fmtDuration(start.untilNow(io, .awake))});
        break :blk try platform.compile(allocator, io, model, .forward, .{input_shape});
    };
    defer exe.deinit();

    // Load weight buffers
    var model_buffers = blk: {
        log.info("Transferring weights...", .{});
        const start: std.Io.Timestamp = .now(io, .awake);
        defer log.info("Transferred weights [{D}]", .{stdx.fmt.fmtDuration(start.untilNow(io, .awake))});
        break :blk try model.load(allocator, io, platform, &store);
    };
    defer Generator.unloadBuffers(&model_buffers);

    // Create input buffer
    var input_buffer: zml.Buffer = try .fromSlice(
        io,
        platform,
        zml.Slice.init(input_shape.shape(), std.mem.sliceAsBytes(img.data)),
    );
    defer input_buffer.deinit();

    // Run inference
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    log.info("Running inference...", .{});
    args.set(.{ model_buffers, input_buffer });
    exe.call(args, &results);

    var output_buffer: zml.Buffer = results.get(zml.Buffer);
    defer output_buffer.deinit();

    // The model output shape may differ from the input when dimensions aren't
    // divisible by 4 (two stride-2 down/upsamples don't roundtrip exactly).
    // Use the actual output dimensions from the buffer.
    const out_shape = output_buffer.shape();
    const out_h: u32 = @intCast(out_shape.dim(2));
    const out_w: u32 = @intCast(out_shape.dim(3));

    // Read output data back to host
    const output_slice = try output_buffer.toSliceAlloc(allocator, io);
    defer output_slice.free(allocator);
    const output_data = output_slice.constItems(f32);

    // Save output image (cropped to original input dimensions)
    log.info("Saving output to {s} ({}x{})...", .{ output_path, img.width, img.height });
    try saveOutputImage(allocator, io, output_path, output_data, out_w, out_h, img.width, img.height);
    log.info("Done!", .{});
}
