import atexit
import cProfile
import inspect
import io
import logging
import pstats
import time
import tracemalloc

from argparse import Action
from pathlib import Path


GLOBAL_TIMER = None
TIME_PROFILER = None
MEMORY_TRACER = None


# == Time-based Profiling == #
class Timer:
    """
    Wrapper class which manages our total runtime timer
    """
    def __init__(self):
        self._t0 = time.time()
        atexit.register(self.stop)

    def __del__(self):
        """
        Prevent early deletion of Timers from breaking atexit
        """
        atexit.unregister(self.stop)

    def stop(self):
        time_delta = time.time() - self._t0
        logging.info(f"Total runtime; {time_delta:.3f} seconds.")


def begin_global_timer():
    # Fetch the PROFILING_TIMER from the global space
    global GLOBAL_TIMER
    # If it hasn't already been set, start the timer
    if GLOBAL_TIMER is None:
        GLOBAL_TIMER = Timer()
    # Otherwise, skip and warn the user
    else:
        logging.warning(
            "Tried to start the global profiling timer twice; you should leave generally leave this initialization to "
            "SCT itself, rather than calling 'begin_global_timer' explicitly!"
        )


class TimeProfilingManager:

    default_output = Path('./time_profiler_results.txt')

    def __init__(self, out_file=None):
        # Initialize and enable the profiler
        self._profiler = cProfile.Profile()
        self._profiler.enable()
        # Track the output file, if it was provided
        if out_file is not None:
            self._output_file = out_file
        else:
            self._output_file = self.default_output
        # Ensure the profiler completes its runtime at program exit
        atexit.register(self._finish_profiling)

    def __del__(self):
        # If the profiler is deleted explicitly, just clean up without reporting anything
        atexit.unregister(self._finish_profiling)
        self._profiler.disable()

    def _finish_profiling(self):
        # Finish profiling
        self._profiler.disable()

        # Ensure the parent folder for the output exists before trying to save it
        self._output_file.parent.mkdir(exist_ok=True, parents=True)

        # Grab the stats, and organize them from most to least time-consuming (cumulatively)
        if self._output_file.suffix == '.prof':
            # If we want binary profiler output, dump the stats to file with the built-in `dump_stats` method
            profiling_stats = pstats.Stats(self._profiler)
            profiling_stats = profiling_stats.sort_stats(pstats.SortKey.CUMULATIVE)
            profiling_stats.dump_stats(self._output_file)
        else:
            # Otherwise, dump the stats in a human-readable format
            io_stream = io.StringIO()
            profiling_stats = pstats.Stats(self._profiler, stream=io_stream)
            profiling_stats = profiling_stats.sort_stats(pstats.SortKey.CUMULATIVE)
            profiling_stats.print_stats()
            with open(self._output_file, 'w') as out_stream:
                out_stream.write(io_stream.getvalue())

        # Report that the file was written, and where to
        logging.info(f"Saved time profiling results to '{self._output_file.resolve()}'.")


def begin_profiling_time(out_path: Path = None):
    # Fetch the PROFILING_TIMER from the global space
    global TIME_PROFILER
    # If it hasn't already been set, initiate and set up the profiler
    if TIME_PROFILER is None:
        # Initialize the profiler
        TIME_PROFILER = TimeProfilingManager(out_path)
    # Otherwise, skip and warn the user
    else:
        logging.warning(
            "Tried to start the time profiler twice; you should leave generally leave this to be handled by the CLI, "
            "rather than calling 'begin_profiling_time' directly!"
        )


class TimeProfilingAction(Action):
    """
    ArgParse action, which initialized the time-based profiler when the argument is present.

    The user can specify either the output directory or file, if they want the results saved somewhere specific.
    If not, the file will be saved in the current working directory (unless the 'default' is explicitly set otherwise)
    """

    default_fname = Path("time_profiling_results.txt")

    def __call__(self, parser, namespace, values, option_string=None):
        # If the length of the values provided is greater than 1, raise an error
        if values is not None and not isinstance(values, str) and len(values) > 1:
            raise ValueError("Only one output file can be specified for profiler outputs!")

        # If no value as provided, set the output path to be in the current directory
        if values is None:
            out_path = Path('.') / self.default_fname
        else:
            # Otherwise, try to get the path associated with this parameter, and confirm its root exists
            out_path = Path(values)

        # Check to see if the file exists and is a directory
        if out_path.exists() and out_path.is_dir():
            # If so, set the result to be saved to our default file output
            out_path /= self.default_fname

        # Finally, initiate the time-based profiler, designating it's output file to be the user specified one
        begin_profiling_time(out_path)

        setattr(namespace, self.dest, out_path)


# == Memory-based Profiling == #
class MemoryTracingManager:

    time_file = Path('peak_memory.txt')
    snapshot_file = Path('memory_snapshots.txt')

    def __init__(self, out_path=Path('.')):
        # Initialize and enable the profiler
        tracemalloc.start()

        # Generate the requested output directory
        out_path.mkdir(parents=True, exist_ok=True)

        # Track the output file destinations
        self._timed_outputs = out_path / self.time_file
        self._snapshot_outputs = out_path / self.snapshot_file

        # Reset the files if they already exist
        if self._timed_outputs.exists():
            self._timed_outputs.unlink()
        if self._snapshot_outputs.exists():
            self._snapshot_outputs.unlink()

        # Function to assess the current memory use
        self._start_time = time.time()

        # Ensure the profiler completes its runtime at program exit
        atexit.register(self._finish_tracing)

    def __del__(self):
        # If the profiler is deleted explicitly, just clean up without reporting anything
        atexit.unregister(self._finish_tracing)
        tracemalloc.stop()

    def snapshot_memory(self, caller) -> Path:
        # Snapshot the current memory state
        mem_snapshot = tracemalloc.take_snapshot()
        mem_stats = mem_snapshot.statistics('lineno', cumulative=True)

        # Generate the header for this output
        header = f"=== {caller.function} ({Path(caller.filename).name}, line {caller.lineno}) ===\n"

        # Creat the initial file if it doesn't already exist
        if not self._snapshot_outputs.exists():
            self._snapshot_outputs.touch()

        # Save the snapshot
        with open(self._snapshot_outputs, 'a') as fp:
            fp.write(header)
            fp.writelines([f'{x}\n' for x in mem_stats])
            fp.write('\n')

    def _finish_tracing(self):
        # Get the peak memory consumed during the program
        _, traced_peak = tracemalloc.get_traced_memory()

        # Finish profiling after sampling (as otherwise the peak is reset to 0)
        tracemalloc.stop()

        # Save the peak results to the output file
        with open(self._timed_outputs, 'x') as ofp:
            ofp.write(f"PEAK; {traced_peak / 1024:.3f}")

        # Report that the file was written, and where to
        logging.info(f"Saved peak memory result to '{self._timed_outputs.resolve()}'.")
        if self._snapshot_outputs.exists():
            logging.info(f"Saved memory snapshot results to '{self._snapshot_outputs.resolve()}'")


def begin_tracing_memory(out_path: Path = None):
    # Fetch the MEMORY_PROFILER from the global space
    global MEMORY_TRACER
    # If it hasn't already been set, initiate and set up the profiler
    if MEMORY_TRACER is None:
        # Initialize the profiler
        MEMORY_TRACER = MemoryTracingManager(out_path)
    # Otherwise, skip and warn the user
    else:
        logging.warning(
            "Tried to start the memory profiler twice; you should leave generally leave this to be handled by the CLI, "
            "rather than calling 'begin_profiling_memory' directly!"
        )


def snapshot_memory():
    # Return early if memory tracing is not active
    if MEMORY_TRACER is None:
        return

    # Get the calling function
    caller = inspect.stack()[1]

    # Snapshot the current memory state, saving it to file immediately and
    # noinspection PyUnresolvedReferences
    MEMORY_TRACER.snapshot_memory(caller)


class MemoryTracingAction(Action):
    """
    ArgParse action, which initializes the memory tracer when the argument is present.

    The user can specify either the output directory or file, if they want the results saved somewhere specific.
    If not, the file will be saved in the current working directory (unless the 'const' is explicitly set otherwise)
    """
    def __call__(self, parser, namespace, values, option_string=None):
        # If the length of the values provided is greater than 1, raise an error
        if values is not None and not isinstance(values, str) and len(values) > 1:
            raise ValueError("Only one output file can be specified for profiler outputs!")

        # If no value as provided, set the output path to be in the current directory
        if values is None:
            out_path = Path('.')
        else:
            # Otherwise, try to get the path associated with this parameter, and confirm its root exists
            out_path = Path(values)

        # Finally, initiate the memory tracer, designating it's output file to be the user specified one
        begin_tracing_memory(out_path)

        setattr(namespace, self.dest, out_path)
