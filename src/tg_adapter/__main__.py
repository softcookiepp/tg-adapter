"""Main"""

# imports: library
from argparse import ArgumentParser
from .testing import run_tests

# imports: dependencies
#import libmonty_logging
#import libmonty_logging.message as logging_message

# imports: project
from . import version


def main() -> None:
	"""Main"""
	"""
	libmonty_logging.apply_default_console_and_file(
		version.PROGRAM_NAME,
		version.__version__
	)

	logging_message.program_header(version.PROGRAM_NAME)
	"""

	parser = ArgumentParser(prog=version.PROGRAM_NAME)

	parser.add_argument('--version',
						help='Display version',
						action='store_true',
						dest='version')
	parser.add_argument("--test", help="Run tests", action = "store_true", dest = "test")

	args = parser.parse_args()

	if args.version:
		print(f'{version.PROGRAM_NAME} {version.__version__}')
		return
	if args.test:
		run_tests()

if __name__ == '__main__':
	main()
