"""
This module generates low-level wrapper code for BLAS functions.

It produces Rust source files that abstract BLAS scalar types and
provide safe access points while preserving the underlying CBLAS
conventions. The generated code should not be edited manually.
"""

import os

from pygments import highlight
from pygments.lexers.rust import RustLexer
from pygments.formatters.terminal256 import Terminal256Formatter

from jinja2 import Environment, FileSystemLoader

import parse_blas_functions as pbf

script_dir = os.path.dirname(os.path.abspath(__file__))

env = Environment(
    loader=FileSystemLoader(os.path.join(script_dir, "templates")), autoescape=False
)

template_scalar = env.get_template("scalar.rs.j2")
template_simple = env.get_template("matmul_simple.rs.j2")

def gen_raw_data():
    data_raw = {
        "imports": [
            "cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_SIDE, CBLAS_UPLO, CBLAS_DIAG}",
            "num_complex::{Complex, ComplexFloat}",
        ],
        "functions_generic": [],
        "functions": {"f32": [], "f64": [], "Complex<f32>": [], "Complex<f64>": []},
        "functions_call": {"f32": [], "f64": [], "Complex<f32>": [], "Complex<f64>": []},
    }
    return data_raw

def add_function_to_data_scalar(data, func):
    """
    Add a function definition to the data structure in generic, Rust, and call formats.
    """

    def build_args(arg_names, arg_types, type_converter):
        return [
            {"name": handle_dot(name), "type": type_converter(arg_type, name, func.name)}
            for name, arg_type in zip(arg_names, arg_types)
        ]

    def handle_dot(argn):
        if "dot" in argn:
            return f"{argn}"
        return argn

    generic_args = build_args(
        func.arg_names, func.arg_types, pbf.convert_c_type_to_generic
    )

    generic_return = None if func.return_type in ("()", None) else func.return_type

    generic_function = {
        "name": func.generic_name,
        "args": generic_args,
        "return_type": pbf.convert_c_type_to_generic(func.return_type, "return", func.name),
    }

    if generic_function not in data["functions_generic"]:
        data["functions_generic"].append(generic_function)

    rust_args = build_args(func.arg_names, func.arg_types, pbf.convert_c_type_to_rust)
    rust_function = {
        "name": func.name,
        "generic_name": func.generic_name,
        "args": rust_args,
        "return_type": pbf.convert_c_type_to_rust(func.return_type, "", ""),
    }
    data["functions"][func.get_rust_type()].append(rust_function)

    call_args = [
        {"name": name, "type": pbf.convert_c_type_for_call(name, arg_type, func.name)}
        for name, arg_type in zip(func.arg_names, func.arg_types)
    ]

    call_function = {
        "name": func.name,
        "args": call_args,
        "return_type": pbf.convert_c_type_to_rust(func.return_type, "", ""),
    }

    data["functions_call"][func.get_rust_type()].append(call_function)

    return data


if __name__ == "__main__":
    lib_file = os.path.join(script_dir, ".", "cblas_sys.rs")
    blas_functions = pbf.parse_lib_rs(lib_file)

    scalar_matmul = gen_raw_data()

    for bf in blas_functions:
        if any(x in bf.name for x in ("gemm", "symm", "trmm", "hemm")):
            scalar_matmul = add_function_to_data_scalar(scalar_matmul, bf)

    output_scalar_matmul = template_scalar.render(**scalar_matmul)

    # print(highlight(output_scalar, RustLexer(), Terminal256Formatter()))

    with open(
        os.path.join(script_dir, "../src/matmul", "scalar.rs"), "w", encoding="utf-8"
    ) as f:
        f.write(output_scalar_matmul)

    scalar_matvec = gen_raw_data()

    functions_for_matvec = ['axpy', 'gemv', 'ger', 'amax', 'dotu', 'dotc', 'nrm2', 'asum', 'copy', 'scal', 'swap', 'symv', 'trmv', 'syr', 'syr2', 'her', 'dot'] #, 'ddot', 'sdot'] # givens rot are missing

    for bf in blas_functions:
        if any(x in bf.name for x in functions_for_matvec) and bf.name != 'dsdot' and bf.name != 'sdsdot':
            scalar_matvec = add_function_to_data_scalar(scalar_matvec, bf)

    output_scalar_matvec = template_scalar.render(**scalar_matvec)

    print(highlight(output_scalar_matvec, RustLexer(), Terminal256Formatter()))

    with open(
        os.path.join(script_dir, "../src/matvec", "scalar.rs"), "w", encoding="utf-8"
    ) as f:
        f.write(output_scalar_matvec)


