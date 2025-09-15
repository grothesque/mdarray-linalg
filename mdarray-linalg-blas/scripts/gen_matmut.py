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
            "num_complex::Complex",
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
            {"name": name, "type": type_converter(arg_type, name, func.name)}
            for name, arg_type in zip(arg_names, arg_types)
        ]

    generic_args = build_args(
        func.arg_names, func.arg_types, pbf.convert_c_type_to_generic
    )
    generic_return = None if func.return_type in ("()", None) else func.return_type

    generic_function = {
        "name": func.generic_name,
        "args": generic_args,
        "return_type": generic_return,
    }

    if generic_function not in data["functions_generic"]:
        data["functions_generic"].append(generic_function)

    rust_args = build_args(func.arg_names, func.arg_types, pbf.convert_c_type_to_rust)
    rust_function = {
        "name": func.name,
        "generic_name": func.generic_name,
        "args": rust_args,
        "return_type": func.return_type,
    }
    data["functions"][func.get_rust_type()].append(rust_function)

    call_args = [
        {"name": name, "type": pbf.convert_c_type_for_call(name, arg_type, func.name)}
        for name, arg_type in zip(func.arg_names, func.arg_types)
    ]

    call_function = {
        "name": func.name,
        "args": call_args,
        "return_type": func.return_type,
    }
    data["functions_call"][func.get_rust_type()].append(call_function)

    return data

def add_function_to_data_simple(data, func):
    def build_args(arg_names, arg_types, type_converter):
        return [
            {"name": name, "type": type_converter(arg_type, name, func.name)}
            for name, arg_type in zip(arg_names, arg_types)
        ]

    data["functions_call_to_scalar"] = func.name

    return data

def mdarray_to_blas(arg_name, arg_type, routine_name=None):
    bat_guessed = pbf.guess_blas_arg_type(arg_name, routine_name)
    # if bat_guessed ==
    if 'layout' in arg_name:
        return 'CBLAS_LAYOUT::CblasRowMajor'
    if 'trans' in arg_name:
        return f'{arg_name[-1]}_trans'
    if 'ld' in arg_name:
        return f'{arg_name[-1]}_stride'
    if bat_guessed == 'array':
        if 'mut' in arg_type:
            return f'{arg_name}.as_mut_ptr() as *mut T'
        return f'{arg_name}.as_ptr()'
    if 'diag' in arg_name:
        return 'CBLAS_DIAG::CblasNonUnit'
    return arg_name

if __name__ == "__main__":
    lib_file = os.path.join(script_dir, ".", "cblas_sys.rs")
    blas_functions = pbf.parse_lib_rs(lib_file)

    data_scalar = gen_raw_data()

    for bf in blas_functions:
        if any(x in bf.name for x in ("gemm", "symm", "trmm", "hemm")):
            data_scalar = add_function_to_data_scalar(data_scalar, bf)

    output_scalar = template_scalar.render(**data_scalar)

    # print(highlight(output_scalar, RustLexer(), Terminal256Formatter()))

    # data_simple = add_function_to_data(

    data_simple = gen_raw_data()
    # for x in ("gemm", "symm", "trmm", "hemm"):
    #     data_simple["functions_generic"].append(x)

    args_gemm = {
        "name": ["alpha", "a", "b", "beta", "c"],
        "type": ["T", "&DSlice<T, 2, La>", "&DSlice<T, 2, Lb>", "T", "&mut DSlice<T, 2, Lc>"],
        "generics": ["T", "La", "Lb", "Lc"],
        "bounds": ["BlasScalar + ComplexFloat", "Layout", "Layout", "Layout"]
    }

    args_symm = {
        "name": ["alpha", "a", "b", "beta", "c", "side", "uplo"],
        "type": ["T", "&DSlice<T, 2, La>", "&DSlice<T, 2, Lb>", "T", "&mut DSlice<T, 2, Lc>", "CBLAS_SIDE", "CBLAS_UPLO"],
        "generics": ["T", "La", "Lb", "Lc"],
        "bounds": ["BlasScalar + ComplexFloat", "Layout", "Layout", "Layout"]
    }

    args_hemm = {
        "name": ["alpha", "a", "b", "beta", "c", "side", "uplo"],
        "type": ["T", "&DSlice<T, 2, La>", "&DSlice<T, 2, Lb>", "T", "&mut DSlice<T, 2, Lc>", "CBLAS_SIDE", "CBLAS_UPLO"],
        "generics": ["T", "La", "Lb", "Lc"],
        "bounds": ["BlasScalar + ComplexFloat", "Layout", "Layout", "Layout"]
    }

    args_trmm = {
        "name": ["alpha", "a", "b", "side", "uplo"],
        "type": ["T", "&DSlice<T, 2, La>", "&mut DSlice<T, 2, Lb>", "CBLAS_SIDE", "CBLAS_UPLO"],
        "generics": ["T", "La", "Lb"],
        "bounds": ["BlasScalar + ComplexFloat", "Layout", "Layout"]
    }
    
    functions_simple = {
        "gemm" : [args_gemm],
        "symm" : [args_symm],
        "hemm" : [args_hemm],
        "trmm" : [args_trmm]
    }

    data_simple["functions_simple"] = functions_simple

    for bf in blas_functions:
        for x in ("gemm", "symm", "trmm", "hemm"):
            if x in bf.name:
                functions_simple[x].append(list(map(lambda x: mdarray_to_blas(x[0],x[1]), zip(bf.arg_names, bf.arg_types))))


    data_simple["init"] = ["","_uninit"]

    output_simple = template_simple.render(**data_simple)

    # print(data_simple)

    print(highlight(output_simple, RustLexer(), Terminal256Formatter()))

    with open(
        os.path.join(script_dir, "../src/matmul", "scalar.rs"), "w", encoding="utf-8"
    ) as f:
        f.write(output_scalar)

    # with open(
    #     os.path.join(script_dir, "../src/matmul", "simple.rs"), "w", encoding="utf-8"
    # ) as f:
    #     f.write(output_simple)
    # Deprecated requires to much handling of specific cases
