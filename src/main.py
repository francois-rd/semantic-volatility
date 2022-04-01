import argparse

from commands import (
    ExistingWordSamplerCommand,
    RedditDownloaderCommand,
    RedditPreprocessorCommand,
    WordUsageFinderCommand,
    BasicDetectorCommand,
    BertCommand,
    TimeSeriesCommand,
    PlotTimeSeriesCommand,
    PlotStatsCommand
)

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(
        description="Main entry point for all programs.")
    subparsers = main_parser.add_subparsers(title="main subcommands")

    # Data gathering and cleaning commands.
    ExistingWordSamplerCommand(subparsers.add_parser(
        'sample-existing', help="randomly samples a number of existing words"))
    RedditDownloaderCommand(subparsers.add_parser(
        'download', help="download a portion of Reddit"))
    RedditPreprocessorCommand(subparsers.add_parser(
        'preprocess', help="preprocess the downloaded Reddit data"))

    # Word usage finding and new word detection commands.
    WordUsageFinderCommand(subparsers.add_parser(
        'find', help='find all usages of each word in the Reddit data'))
    BasicDetectorCommand(subparsers.add_parser(
        'basic-detect', help='detect new words from simple time slice cutoffs'))

    # Modeling commands.
    BertCommand(subparsers.add_parser(
        'bert', help='embed detected new words with a BERT-like model'))
    TimeSeriesCommand(subparsers.add_parser(
        'time-series', help='compute multiple time series from embeddings'))

    # Analysis commands.
    PlotTimeSeriesCommand(subparsers.add_parser(
        'plot-ts', help='plot the multiple resulting time series'))
    PlotStatsCommand(subparsers.add_parser(
        'plot-stats', help='compute and plot stats from the time series'))

    args = main_parser.parse_args()
    args.func(args)
