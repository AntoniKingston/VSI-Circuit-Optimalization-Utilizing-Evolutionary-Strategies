import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import array
import sys
import logging
from typing import Any, Optional

import matlab.engine
from xmlrpc.server import SimpleXMLRPCServer

_MATLAB_SOURCE_PATH = "."
_LOG = logging.getLogger("VSI-SRV")


@dataclass
class SimulatorConfiguration:
    Lf: float = 1e-3
    Cf: float = 22e-6
    Rf: float = 35e-3
    fsw: float = 20e3
    f: float = 50.0
    Vdc: float = 700.0
    Tsim: float = 0.3
    Vref: float = 100.0
    t_settling: float = 40e-3
    beta_c: float = 3e-5

    @classmethod
    def from_dict(cls, input: dict[str, Any]):
        return cls(
            input["Lf"],
            input["Cf"],
            input["Rf"],
            input["fsw"],
            input["f"],
            input["Vdc"],
            input["Tsim"],
            input["Vref"],
            input["t_settling"],
            input["beta_c"],
        )


class DLQREvaluator:
    # def __init__(self, sim_cfg: Optional[SimulatorConfiguration] = None) -> None:
    #     if not sim_cfg:
    #         sim_cfg = SimulatorConfiguration()
    #         _LOG.info(f"VSI simulator will use default parameters: [{sim_cfg}].")
    #     else:
    #         _LOG.info(f"VSI simulator will use custom parameters: {sim_cfg}")
    #     self._matlab_engine = matlab.engine.start_matlab()
    #     self._matlab_engine.cd(_MATLAB_SOURCE_PATH)
    #     self._simulator_params = self._matlab_engine.SimulatorParameters(
    #         sim_cfg.Lf,
    #         sim_cfg.Cf,
    #         sim_cfg.Rf,
    #         sim_cfg.fsw,
    #         sim_cfg.f,
    #         sim_cfg.Vdc,
    #         sim_cfg.Tsim,
    #         sim_cfg.Vref,
    #         sim_cfg.t_settling,
    #         sim_cfg.beta_c,
    #     )
    #     self._simulator = self._matlab_engine.Simulator(self._simulator_params)
    #     self._dlqr_evaluator = self._matlab_engine.DLQREvaluator(self._simulator)
    def __init__(self, sim_cfg: Optional[SimulatorConfiguration] = None) -> None:
        if not sim_cfg:
            sim_cfg = SimulatorConfiguration()
            _LOG.info(f"VSI simulator will use default parameters: [{sim_cfg}].")
        else:
            _LOG.info(f"VSI simulator will use custom parameters: {sim_cfg}")

        _LOG.info("Connecting to shared MATLAB engine 'ubuntu_matlab'...")
        self._matlab_engine = matlab.engine.connect_matlab("ubuntu_matlab")
        self._matlab_engine.addpath(
        "/home/antek/studia/MSI/wdae/VSI-Circuit-Optimalization-Utilizing-Evolutionary-Strategies/src/matlab",
        nargout=0
        )

        _LOG.info("Connected to MATLAB engine.")

        # self._matlab_engine.cd(_MATLAB_SOURCE_PATH)

        self._simulator_params = self._matlab_engine.SimulatorParameters(
            sim_cfg.Lf,
            sim_cfg.Cf,
            sim_cfg.Rf,
            sim_cfg.fsw,
            sim_cfg.f,
            sim_cfg.Vdc,
            sim_cfg.Tsim,
            sim_cfg.Vref,
            sim_cfg.t_settling,
            sim_cfg.beta_c,
        )

        self._simulator = self._matlab_engine.Simulator(self._simulator_params)
        self._dlqr_evaluator = self._matlab_engine.DLQREvaluator(self._simulator)

    def evaluate(self, input: Sequence[float]) -> float:
        return self._matlab_engine.evaluate(
            self._dlqr_evaluator, array.array("d", input)
        )

    def evaluate_batch(self, input: Sequence[Sequence[float]]) -> list[float]:
        return list(map(lambda e: self.evaluate(e), input))


def _load_json_from_fs(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Path {path} does not exist.")
    with path.open() as f:
        js = json.loads(f.read())
        return js


def main():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        prog="VSI circuit simulator with DLQR evaluator XML-RPC server.",
    )
    parser.add_argument(
        "-a", "--address", default="127.0.0.1", type=str, help="Server IPv4 address."
    )
    parser.add_argument(
        "-p", "--port", default=8484, type=int, help="Server IPv4 address port."
    )
    parser.add_argument(
        "-c", "--configuration", type=Path, help="Filepath to simulator configuration."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display XML-RPC server request logs on STDOUT",
    )
    args = parser.parse_args()
    display_logs = True if args.verbose else False
    cfg = (
        SimulatorConfiguration.from_dict(_load_json_from_fs(args.configuration))
        if args.configuration
        else None
    )
    with SimpleXMLRPCServer((args.address, args.port), logRequests=display_logs) as srv:
        _LOG.info(
            f"Starting VSI circuit simulator with DLQR evaluator XML-RPC server on address {args.address}:{args.port}."
        )

        srv.register_introspection_functions()
        srv.register_instance(DLQREvaluator(cfg))
        srv.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"Fatal error occured. Shutting down VSI simulator. Error: {err}")
        sys.exit(-1)
