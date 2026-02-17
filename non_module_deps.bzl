load("//third_party/arocc:repo.bzl", arocc = "repo")
load("//third_party/translate-c:repo.bzl", translate_c = "repo")

def _non_module_deps_impl(mctx):
    arocc()
    translate_c()

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

non_module_deps = module_extension(
    implementation = _non_module_deps_impl,
)
