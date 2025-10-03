"""
BLAS function parser for Rust bindings.

This module parses CBLAS function signatures from Rust extern "C" blocks and creates
higher-level Python objects that facilitate code generation around these functions.
It handles type inference, signature parsing, and conversion between C and Rust types.
"""

import sys
import re
from pathlib import Path

def guess_blas_arg_type(arg_name, routine_name=None):
    """
    Infer the BLAS argument type from its name and (optionally) the routine name.
    """
    arg = arg_name.lower()
    routine = routine_name.lower() if routine_name else ""

    # Using sets for O(1) membership testing and to prevent duplicate entries
    integers = {"m", "n", "k", "lda", "ldb", "ldc", "incx", "incy", "ku", "kl"}
    arrays   = {"x", "y", "a", "b", "c", "ap", "bp", "cp"}
    scalars  = {"alpha", "beta"}
    cblas    = {"layout", "uplo", "diag", "side", "trans", "transa", "transb"}

    # --- Special cases by routine ---
    special_cases = [
        # Hermitian routines
        (("her2k","her2"), {"alpha": "scalar_of_data_type", "beta": "real_scalar"}),
        ("herk",  {s: "real_scalar" for s in scalars}),
        ("her", {"alpha": "real_scalar"}),
        ("hemm",  {s: "scalar_of_data_type" for s in scalars}),
        # Dot products
        (("dotu_sub", "dotc_sub", "dot"), {"dotu": "scalar_of_data_type", "dotc": "scalar_of_data_type"}),
        (("cdotu", "cdotc", "zdotu", "zdotc"), {"pres": "scalar_of_data_type"}),
        # Complex abs
        ("cabs1", {"z": "scalar_of_data_type", "c": "scalar_of_data_type"}),
        # Real dot products
        (("sdsdot", "dsdot"), {"sb": "scalar_of_data_type"}),
        # Real scale of a complex vector
        (("zdscal", "csscal"), {"alpha": "real_scalar"}),
        (("sscal", "dscal", "cscal", "zscal"), {"alpha": "scalar_of_data_type"}),
    ]

    for keys, mapping in special_cases:
        if isinstance(keys, str):
            if keys in routine and arg in mapping:
                return mapping[arg]
        else:  # tuple of possible substrings
            if any(k in routine for k in keys) and arg in mapping:
                return mapping[arg]

    # --- Rotation routines ---
    if any(r in routine for r in ("rotg", "rotmg", "rot", "rotm")):
        if arg in {"a", "b", "d1", "d2", "x1", "y1", "b1", "b2", "s"}:
            return "scalar_of_data_type"
        if arg == "c":
            return "real_scalar"
        if arg in {"param", "p"}:
            return "array"
    # --- Standard cases ---
    if arg in cblas:
        return "cblas_option"
    if arg in integers:
        return "integer"
    if arg in arrays:
        return "array"
    if arg in scalars:
        return "scalar_of_data_type"

    return "unknown"


class BlasFunction:
    """
    Represents a parsed BLAS function with type information and signature details.

    Extracts BLAS prefix (s/d/c/z), operation name, and argument details from
    function signatures to enable generic code generation.
    """

    BLAS_PREFIXES = {'s': 'float', 'd': 'double', 'c': 'complex<float>', 'z': 'complex<double>'}

    def __init__(self, name, signature):
        """Initialize BlasFunction by parsing name and signature."""
        self.name = name.replace("cblas_", "")

        self.generic_name = "cblas_" + self.name[1:]
        self.signature = signature.replace("cblas_", "")

        m = re.match(r'^cblas_([sdcz])([a-zA-Z0-9]+)$', name)

        if "scal" in self.name:
            if len(self.name) == 6:
                self.generic_name = "cblas_" + "r" + self.name[2:]
            self.prefix = self.name[0]

        elif any(op in self.name for op in ("nrm2", "asum")):
            mapping = {
                "sn": ("s", "s"),
                "sa": ("s", "s"),
                "dn": ("d", "d"),
                "da": ("d", "d"),
                "sc": ("c", "s"),
                "dz": ("z", "d"),
            }

            key = self.name[:2]
            if key not in mapping:
                raise ValueError(f"Unsupported prefix for {self.name}")

            self.prefix, generic_prefix = mapping[key]

            if len(self.name) == 5:
                self.generic_name = "cblas_" + self.name[1:]
            else:
                self.generic_name = "cblas_" + self.name[2:]

            self.operation = name.rstrip('_')
        elif m:
            self.prefix = m.group(1)
            self.operation = m.group(2)
        elif "dot" in self.name:
            self.prefix = self.name[0]
            self.operation = name.rstrip('_')

        else:
            self.prefix = None
            self.operation = name.rstrip('_')

        if "max" in self.name:
            self.prefix = self.name[1]
            self.name = "argmax"


        self.blas_type = self.BLAS_PREFIXES.get(self.prefix, None)

        self.arg_types = []
        self.arg_names = []
        self.return_type = '()'
        self.parse_signature(self.signature)

    def parse_signature(self, signature):
        """Parse Rust function signature to extract argument names, types and return type."""
        # remove whitespaces and \n
        clean_sig = ' '.join(signature.split())

        args_match = re.search(r'fn\s+\w+\s*\((.*?)\)\s*(?:->\s*([^;]+))?', clean_sig)

        args = []

        if args_match:
            args_str = args_match.group(1).strip()
            self.return_type = args_match.group(2).strip() if args_match.group(2) else '()'

            if args_str:
                current_arg = ""
                paren_level = 0

                for char in args_str + ',':  #
                    if char == ',' and paren_level == 0:
                        if current_arg.strip():
                            args.append(current_arg.strip())
                        current_arg = ""
                    else:
                        if char in '(<[':
                            paren_level += 1
                        elif char in ')>]':
                            paren_level -= 1
                        current_arg += char

                self.arg_types = []
                for arg in args:
                    if ':' in arg:
                        arg_type = arg.split(':', 1)[1].strip()
                        self.arg_names.append(arg.split(':', 1)[0].strip())
                    else:
                        print(f"Error, can't parse {arg}. Abort.")
                        sys.exit(1)

                    self.arg_types.append(arg_type)

    def get_rust_type(self, prefix=None):
        """Get the corresponding Rust primitive type for the BLAS prefix."""
        type_map = {
            's': 'f32',
            'd': 'f64',
            'c': 'Complex<f32>',
            'z': 'Complex<f64>',
        }
        if prefix:
            return type_map.get(prefix, 'f64')
        return type_map.get(self.prefix, 'f64')

    def __repr__(self):
        return f"<BlasFunction {self.name} prefix={self.prefix} type={self.blas_type} operation={self.operation}>"

def parse_lib_rs(lib_path):
    """Parse lib.rs to extract all extern 'C' function signatures."""
    content = Path(lib_path).read_text(encoding="utf-8")

    # Remove comments
    content = re.sub(r'//.*?\n', '\n', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    # Extract extern "C" blocks
    extern_blocks = re.findall(r'extern\s*"C"\s*{(.*?)}', content, re.DOTALL)
    functions = []

    for block in extern_blocks:
        normalized_block = ' '.join(block.split())

        function_pattern = r'pub\s+fn\s+(\w+_?\w*)\s*\((.*?)\)\s*(?:->\s*([^;]+?))?\s*;'

        for match in re.finditer(function_pattern, normalized_block):
            name = match.group(1)
            args = match.group(2)
            return_type = match.group(3) if match.group(3) else '()'

            signature = f'pub fn {name}({args})'
            if return_type != '()':
                signature += f' -> {return_type}'
            signature += ';'

            functions.append(BlasFunction(name, signature))

    return functions

def convert_c_type_to_rust(c_type, arg_name, routine_name):
    """Convert C type from BLAS signature to Rust type."""
    bat_guessed = guess_blas_arg_type(arg_name, routine_name=routine_name)
    c_type = c_type.strip()

    # Map C types to Rust types
    type_mapping = {
        '*const c_int': 'i32',
        '*mut c_int': '*mut i32',
        '*const c_float': '*const f32',
        '*mut c_float': '*mut f32',
        '*const c_double': '*const f64',
        '*mut c_double': '*mut f64',
        '*const c_float_complex': '*const [f32;2]',
        '*mut c_float_complex': '*mut Complex<f32>',
        '*const c_double_complex': '*const [f64;2]',
        '*mut c_double_complex': '*mut Complex<f64>',
        'c_int': 'i32',
        'c_float': 'f32',
        'c_double': 'f64',
    }

    if "dot" in arg_name:
        return '*mut Complex<f32>' if "float" in c_type  else '*mut Complex<f64>'

    # if "dot" in routine_name and

    if bat_guessed=="array":
        if '*const' in c_type:
            base_type = c_type.replace('*const ', '')
            if "complex" in base_type:
                return '*const Complex<f32>' if "float" in base_type  else '*const Complex<f64>'
            return '*const f32' if "float" in base_type  else '*const f64'
        if '*mut' in c_type:
            base_type = c_type.replace('*mut ', '')
            if "complex" in base_type:
                return '*mut Complex<f32>' if "float" in base_type  else '*mut Complex<f64>'
            return '*mut f32' if "float" in base_type  else '*mut f64'

    if bat_guessed=="scalar_of_data_type":
        if "mut" in c_type:
            if "complex" in c_type:
                return 'mut Complex<f32>' if "float" in c_type  else 'mut Complex<f64>'
            return 'mut f32' if "float" in c_type  else 'mut f64'

        if "complex" in c_type:
            return 'Complex<f32>' if "float" in c_type  else 'Complex<f64>'
        return 'f32' if "float" in c_type  else 'f64'

    return type_mapping.get(c_type, c_type)

def convert_c_type_to_generic(c_type, param_name, routine_name):
    """Generate appropriate casting expression for calling C function from Rust."""
    bat_guessed = guess_blas_arg_type(param_name, routine_name=routine_name)

    generic_type = ''

    if ("nrm2" in routine_name or "asum" in routine_name) and param_name == "return": # should be handled elsewhere
        generic_type = "Self::Real"

    elif ("sdot" in routine_name or "ddot" in routine_name) and param_name == "return":
        generic_type = "Self"

    elif "dot" in param_name:
        generic_type = "*mut Self"

    elif bat_guessed == "array":
        ptr_kind = "mut" if "mut" in c_type else "const"
        generic_type = f"*{ptr_kind} Self"

    elif bat_guessed == "real_scalar":
        generic_type = "Self::Real"

    elif bat_guessed == "scalar_of_data_type":
        generic_type = "Self"

    elif 'c_int' in c_type:
        generic_type = "i32"

    else:
        generic_type = f"{convert_c_type_to_rust(c_type, param_name, routine_name)}"

    return generic_type

def convert_c_type_for_call(arg_name, arg_type, routine_name):
    """Convert C type to generic Rust type using Self and Self::Real."""
    bat_guessed = guess_blas_arg_type(arg_name, routine_name=routine_name)

    cast_call = ''

    match bat_guessed:
        case "integer":
            cast_call = f"{arg_name}"

        case "array":
            ptr_kind = "*mut" if "mut" in arg_type else "*const"
            cast_call = f"{arg_name} as {ptr_kind} _"

        case "scalar_of_data_type":
            if "complex" in arg_type:
                if "dot" in arg_name:
                    cast_call = f"{arg_name} as *mut _"
                elif "mut" in arg_type:
                    cast_call = f"&mut {arg_name} as *mut _ as *mut _"
                else:
                    cast_call = f"&{arg_name} as *const _ as *const _"
            else:
                rust_ty = convert_c_type_to_rust(arg_type, arg_name, routine_name)
                cast_call = f"{arg_name}"

        case "real_scalar":
            if "float" in arg_type:
                cast_call = f"{arg_name}"
            elif "double" in arg_type:
                cast_call = f"{arg_name}"
            else:
                cast_call = f"{arg_name}"

        case "cblas_option":
            cast_call = f"{arg_name}"

        case "unknown":
            print(f"Can't guess arg_type for {arg_name}: {arg_type} in {routine_name}")
            sys.exit(1)

    return cast_call
