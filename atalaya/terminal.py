import os
import time
from datetime import timedelta

import psutil

# Mapping of common color names to their ANSI escape codes
NAMED_COLORS = {
    "reset": "\033[0m",
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_black": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
}

CURSOR_UP = "\033[1A"
CLEAR = "\x1b[2K"


class Terminal:
    def __init__(
        self,
        log_file=None,
        error_color="red",
        ok_color="green",
        warning_color="yellow",
        info_color="cyan",
        reset_color="reset",
    ):
        self.log_file = log_file
        # Use named colors by default
        self.colors = {
            "error": self.get_color_code(error_color),
            "ok": self.get_color_code(ok_color),
            "warning": self.get_color_code(warning_color),
            "info": self.get_color_code(info_color),
            "reset": self.get_color_code(reset_color),
        }

        self._printed_lines = 0

        # get program starting time
        current_process = psutil.Process()  # get current process
        self._start_time = current_process.create_time()  # get process start time

    def _count_lines(self, *message):
        total_lines = 0
        for msg in message:
            total_lines += str(msg).count("\n") + 1
        return total_lines

    def _get_timestamp(self):
        elapsed_time = timedelta(seconds=time.time() - self._start_time)

        # Convert to string and replace " days, " with "-"
        formatted_time = str(elapsed_time).replace(" days, ", "-")
        if "-" not in formatted_time:
            formatted_time = "0-" + formatted_time

        # If milliseconds are present, format them to 3 decimal places
        if "." in formatted_time:
            formatted_time = formatted_time[
                : formatted_time.index(".") + 4
            ]  # Keep only 2 digits after "."

        return formatted_time

    def set_named_color(self, name, ansi_code):
        """Register or overwrite a named ANSI color code for reuse."""
        if not isinstance(name, str) or not name:
            raise ValueError("Color name must be a non-empty string.")
        if not isinstance(ansi_code, str) or not ansi_code.startswith("\033"):
            raise ValueError(
                "ANSI color codes must be strings starting with the escape sequence '\\033'."
            )
        NAMED_COLORS[name] = ansi_code

    def set_log_file(self, log_file):
        self.log_file = log_file

    def get_color_code(self, color):
        """Return the corresponding ANSI code for a color name or check if it is a valid terminal color code."""
        if color in NAMED_COLORS:
            return NAMED_COLORS[color]
        elif isinstance(color, str) and color.startswith("\033"):
            return color  # Assume it's already a valid ANSI escape code
        else:
            raise ValueError(
                f"Invalid color: '{color}'. Use a valid color name or terminal color code."
            )

    def print_message(
        self,
        *message,
        with_timestamp=True,
        message_type=None,
        color=None,
        end="\n",
    ):
        color_override = color
        if message_type is None:
            color_code = self.get_color_code(color_override or "reset")
        else:
            color_code = self.colors.get(message_type, self.colors["reset"])
            if color_override is not None:
                color_code = self.get_color_code(color_override)

        if with_timestamp:
            message = [f"[{self._get_timestamp()}]", *message]

        print(color_code, *message, self.colors["reset"], end=end)
        self._printed_lines += self._count_lines(*message)

        if self.log_file is not None:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "a+") as f:
                f.write(" ".join(map(str, message)) + "\n")

        if message_type == "error":
            raise RuntimeError(" ".join(map(str, message)))

    def set_color(self, message_type, color):
        """Set custom color codes or named colors for a specific message type."""
        if message_type in self.colors:
            self.colors[message_type] = self.get_color_code(color)
        else:
            raise ValueError(f"Invalid message type: {message_type}")

    # Convenient methods for specific types of messages
    def print_error(self, *message, with_timestamp=True, color=None, end="\n"):
        self.print_message(
            "ERROR:",
            *message,
            with_timestamp=with_timestamp,
            message_type="error",
            color=color,
            end=end,
        )

    def print_ok(self, *message, with_timestamp=True, color=None, end="\n"):
        self.print_message(
            *message,
            with_timestamp=with_timestamp,
            message_type="ok",
            color=color,
            end=end,
        )

    def print_warning(self, *message, with_timestamp=True, color=None, end="\n"):
        self.print_message(
            *message,
            with_timestamp=with_timestamp,
            message_type="warning",
            color=color,
            end=end,
        )

    def print_info(self, *message, with_timestamp=True, color=None, end="\n"):
        self.print_message(
            *message,
            with_timestamp=with_timestamp,
            message_type="info",
            color=color,
            end=end,
        )

    def clean(self, number_of_lines="all"):
        if number_of_lines == "all":
            number_of_lines = self._printed_lines
        else:
            number_of_lines = int(number_of_lines)

        for _ in range(number_of_lines):
            print(f"{CURSOR_UP}{CLEAR}", end="")
            self._printed_lines -= 1


# Create an instance of the Terminal class
terminal = Terminal()
