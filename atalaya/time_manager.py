import time

from .terminal import terminal


def format_time(seconds, time_format="%d-%H:%M:%S.%f", floating_precision=3):
    """
    Format a time in seconds into a string according to a given time format.

    Args:
        seconds (float): The time in seconds.
        time_format (str): The format string for the output.
        floating_precision (int): The number of decimals to use for the last part.

    Returns:
        str: The formatted time string.

    Example time formats:
        "%d days, %H:%M:%S.%f"
        "%H:%M:%S.%f"
        "%H"
        "%M"
        "%S"
    """

    # Mapping of time components to their placeholders, units, and precision
    time_components = {
        "milliseconds": ("%f", 1000, 3),
        "seconds": ("%S", 60, 1),
        "minutes": ("%M", 60, 1),
        "hours": ("%H", 24, 1),
        "days": ("%d", 1, 1),
    }

    # Precomputed total values for each time component
    time_totals = {
        "milliseconds": seconds * 1000,
        "seconds": seconds,
        "minutes": seconds / 60,
        "hours": seconds / 3600,
        "days": seconds / (24 * 3600),
    }

    # Track how many format specifiers are in the time format
    total_placeholders = time_format.count("%")

    for time_unit, (placeholder, unit_limit, precision) in time_components.items():
        remaining_placeholders = time_format.count("%")
        if remaining_placeholders == 0:
            break

        if placeholder in time_format:
            total_value = time_totals[time_unit]

            if total_placeholders == 1:  # If it is the only element
                formatted_value = f"{total_value:.0{floating_precision}f}"
            elif remaining_placeholders == 1:  # If it's the last element
                formatted_value = f"{int(total_value):0{precision}d}"
            else:
                formatted_value = f"{int(total_value % unit_limit):0{precision}d}"

            time_format = time_format.replace(placeholder, formatted_value)

    if time_format.count("%") != 0:
        raise ValueError("The given time format is not valid.")

    return time_format


class Timer:
    def __init__(self, name="default_timer", disabled=False, display_color="magenta"):
        """Initialize a new Timer instance."""
        self.name = name
        self.disabled = disabled
        self.start_time: float | None = None
        self.stop_time: float | None = None
        self.total_elapsed_time: float = 0.0  # Total time accumulated across all runs
        self.run_durations: list[float] = []  # History of each run's duration
        self.is_active: bool = False  # Tracks whether the timer is currently running
        self.display_color = display_color  # Color for printed reports

    def start(self) -> "Timer":
        """Start the timer. Raises an error if already running."""
        if self.disabled:
            return self

        if self.is_active:
            raise RuntimeError(f"Timer '{self.name}' is already running.")
        self.start_time = time.time()
        self.is_active = True
        return self

    def stop(self) -> "Timer":
        """Stop the timer and record the duration. Raises an error if not running."""
        if self.disabled:
            return self

        if not self.is_active:
            raise RuntimeError(f"Timer '{self.name}' is not running.")
        assert self.start_time is not None, f"Timer '{self.name}' was never started!"

        # Calculate and record the run duration
        self.stop_time = time.time()
        elapsed_time = self.stop_time - self.start_time
        self.total_elapsed_time += elapsed_time
        self.run_durations.append(elapsed_time)

        # Reset for the next run
        self.start_time = None
        self.is_active = False
        return self

    def is_running(self) -> bool:
        """Check if the timer is currently running."""
        if self.disabled:
            return False
        return self.is_active

    def report(self, report_type="total", time_format="%d-%H:%M:%S.%f"):
        """
        Generate a report for the timer's total elapsed time or statistics across multiple runs.

        Args:
            report_type (str): 'total' for just total time, 'total_with_stats' for min, max, avg.
            time_format (str): Format string for displaying time.
        """
        if self.disabled:
            return

        if self.is_active:
            # If timer is still running, calculate the total including the current run
            total_time = time.time() - self.start_time
        else:
            total_time = self.total_elapsed_time

        formatted_total_time = format_time(total_time, time_format)

        # if there are no runs, just report the total time
        if len(self.run_durations) == 0 and report_type == "total_with_stats":
            report_type = "total"

        if report_type == "total":
            terminal.print_message(
                f"[{self.name}] Total Time: {formatted_total_time}",
                color=self.display_color,
            )
        elif report_type == "total_with_stats":
            min_duration = format_time(min(self.run_durations), time_format)
            max_duration = format_time(max(self.run_durations), time_format)
            avg_duration = format_time(
                sum(self.run_durations) / len(self.run_durations), time_format
            )
            terminal.print_message(
                f"[{self.name}] Total: {formatted_total_time} | Runs: {len(self.run_durations)}, "
                f"Min: {min_duration}, Max: {max_duration}, Avg: {avg_duration}",
                color=self.display_color,
            )
        else:
            raise ValueError(f"Invalid report type: '{report_type}'")

    def reset(self):
        """Reset the timer to its initial state."""
        if self.disabled:
            return

        self.start_time = None
        self.stop_time = None
        self.total_elapsed_time = 0.0
        self.run_durations = []
        self.is_active = False

    def decorator(self):
        """Decorator to time a function."""
        if self.disabled:
            return lambda func: func  # Return original function without timing

        def decorator(func):
            def wrapper(*args, **kwargs):
                self.start()
                result = func(*args, **kwargs)
                self.stop()
                return result

            return wrapper

        return decorator

    def __enter__(self):
        if not self.disabled:
            self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.disabled:
            self.stop()
        if exc_type:
            return False
        return self


class TimeManager:
    def __init__(
        self,
        disabled=False,
        color="magenta",
        time_format="%d-%H:%M:%S.%f",
        report_type="total",
    ):
        """
        Initialize a TimeManager instance to manage multiple timers and generate reports.

        Args:
            disabled (bool): Whether to disable all timers.
            color (str): The default color for timer reports.
            time_format (str): The default format for time display.
            report_type (str): The default report type ('total' or 'total_with_stats').
        """
        self.disabled = disabled
        self.default_color = color
        self.default_time_format = time_format
        self.default_report_type = report_type
        self.timers_dict = {}  # Dictionary to store all timers

    def create_timer(self, timer_name, color=None):
        """Create a new timer with a specified name."""
        assert (
            timer_name not in self.timers_dict
        ), f"Timer '{timer_name}' already exists!"
        timer_color = color if color is not None else self.default_color
        self.timers_dict[timer_name] = Timer(
            name=timer_name, disabled=self.disabled, display_color=timer_color
        )

    def get_timer(self, timer_name):
        """Retrieve an existing timer by name."""
        assert timer_name in self.timers_dict, f"Timer '{timer_name}' does not exist!"
        return self.timers_dict[timer_name]

    def start_timer(self, timer_name):
        """Start a timer by name."""
        self.get_timer(timer_name).start()

    def stop_timer(self, timer_name):
        """Stop a timer by name."""
        self.get_timer(timer_name).stop()

    def report(self, timer_name=None, time_format=None, report_type=None):
        """
        Generate and print a report for a specific timer or all timers.

        Args:
            timer_name (str, optional): The name of the timer to report on. If None, report on all timers.
            time_format (str, optional): Custom time format to use for the report.
            report_type (str, optional): The type of report ('total' or 'total_with_stats').
        """
        if self.disabled:
            return

        time_format = time_format or self.default_time_format
        report_type = report_type or self.default_report_type

        if timer_name:
            # Report for a single timer
            self.get_timer(timer_name).report(
                time_format=time_format, report_type=report_type
            )
        else:
            # Report for all timers
            terminal.print_message(
                "------------ Time Report ------------", color=self.default_color
            )
            for timer in self.timers_dict.values():
                timer.report(time_format=time_format, report_type=report_type)
            terminal.print_message(
                "-------------------------------------", color=self.default_color
            )

    def reset_timer(self, timer_name=None):
        """Reset a specific timer or all timers to their initial state."""
        if self.disabled:
            return
        if timer_name:
            self.get_timer(timer_name).reset()
        else:
            for timer in self.timers_dict.values():
                timer.reset()

    def remove_timer(self, timer_name=None, print_report_before_removal=False):
        """
        Remove a timer from the report. Optionally print a report before removal.

        Args:
            timer_name (str, optional): The name of the timer to remove. If None, remove all timers.
            print_report_before_removal (bool): Whether to print a report before removing the timer.
        """
        if self.disabled:
            return
        if timer_name:
            if print_report_before_removal:
                self.get_timer(timer_name).report()
            del self.timers_dict[timer_name]
        else:
            if print_report_before_removal:
                self.report()
            self.timers_dict.clear()

    def decorator(self, timer_name):
        """Decorator to time a function."""
        if self.disabled:
            return lambda func: func  # Return the original function if disabled
        return self.get_timer(timer_name).decorator()

    def context(self, timer_name):
        """Return a context manager for a timer to automatically handle start and stop."""
        if self.disabled:
            return None  # Return None if disabled
        return self.get_timer(timer_name)

    def __call__(self, timer_name):
        """
        Return either a context manager or a decorator based on the usage.
        If timer_name is not in the dictionary, create a new timer automatically.

        Args:
            timer_name (str): The name of the timer.
        """

        if timer_name not in self.timers_dict:
            self.create_timer(timer_name)

        timer = self.get_timer(timer_name)
        return _TimerWrapper(timer)


class _TimerWrapper:
    """This class can act as both a decorator and a context manager."""

    def __init__(self, timer):
        self.timer = timer

    # Context management methods
    def __enter__(self):
        if self.timer.disabled:
            return None
        return self.timer.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.timer.disabled:
            return None
        return self.timer.__exit__(exc_type, exc_value, traceback)

    # Make the object callable to support decorator usage
    def __call__(self, func):
        if self.timer.disabled:
            return func  # Return original function if disabled
        return self.timer.decorator()(func)
