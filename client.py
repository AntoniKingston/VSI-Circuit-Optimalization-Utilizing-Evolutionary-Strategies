import xmlrpc.client
import argparse

# Currently, MATLAB's implementation hardcodes a number of decision variables.
_PROBLEM_DIMENSIONALITY = 4


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="A CLI client to the VSI simulator/evaluator XML-RPC server.",
    )
    parser.add_argument(
        "-a",
        "--address",
        default="127.0.0.1",
        type=str,
        help="XML-RPC Server IPv4 address.",
    )
    parser.add_argument(
        "-p", "--port", default=8484, type=int, help="XML-RPC Server IPv4 address port."
    )
    parser.add_argument(
        "-d",
        "--data",
        nargs="+",
        required=True,
        help=f"Array of {_PROBLEM_DIMENSIONALITY} real numbers",
    )
    args = parser.parse_args()
    input_data = [float(e) for e in args.data]
    if len(input_data) != _PROBLEM_DIMENSIONALITY:
        raise RuntimeError(
            f"Input data vector size is {_PROBLEM_DIMENSIONALITY}. Given size: {len(input_data)}."
        )
    client = xmlrpc.client.ServerProxy(f"http://{args.address}:{args.port}")
    objective_fn_value = client.evaluate(input_data)
    print(f"Objective function value = {objective_fn_value} (input: {input_data}).")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"XML-RCP client failed. Details: {err}")
