#!/usr/bin/env python
# Copyright (C) 2015-2019 Splunk Inc. All Rights Reserved.
from exec_anaconda import exec_anaconda_or_die
from splunk_path_util import get_mltk_bin_path

import sys

exec_anaconda_or_die()

sys.path.append(get_mltk_bin_path())

with open("/Users/gburgett/Downloads/frank.txt", "w") as ofile:
    ofile.write(str(sys.path))

import os
import conf


from util.param_util import is_truthy, parse_args, convert_params
from util.telemetry_util import log_uuid, log_fit_time, log_app_details, Timer
from util import command_util

import cexc
from chunked_controller import ChunkedController
from cexc import BaseChunkHandler, CommandType

logger = cexc.get_logger('forecast')
messages = cexc.get_messages_logger()

class ForecastCommand(cexc.BaseChunkHandler):
    """FitCommand uses ChunkedController & one of two processors to fit models.

    The FitCommand can use either the FitBatchProcessor or the FitPartialProcessor,
    which is chosen based on the presence of the partial_fit parameter.
    """

    @staticmethod
    def handle_arguments(getinfo):
        """Take the getinfo metadata and return controller_options.

        Args:
            getinfo (dict): getinfo metadata from first chunk

        Returns:
            controller_options (dict): options to be passed to controller
            partial_fit (bool): boolean flag to indicate partial fit
        """
        if len(getinfo['searchinfo']['raw_args']) == 0:
            raise RuntimeError('First argument must be an "algorithm"')

        raw_options = parse_args(getinfo['searchinfo']['raw_args'][1:])
        controller_options, partial_fit = FitCommand.handle_raw_options(raw_options)
        controller_options['algo_name'] = getinfo['searchinfo']['args'][0]

        log_app_details(getinfo['searchinfo'].get('app'))
        return controller_options, partial_fit

    @staticmethod
    def handle_raw_options(controller_options):
        """Load command specific options.

        Args:
            controller_options (dict): options from handle_arguments
        Returns:
            controller_options (dict): dict of controller options
            partial_fit (dict): boolean flag for partial fit
        """
        controller_options['processor'] = 'FitBatchProcessor'
        partial_fit = False

        if 'params' in controller_options:
            try:
                fit_params = convert_params(
                    params=controller_options['params'],
                    ignore_extra=True,
                    bools=['apply', 'partial_fit'],
                )
            except ValueError as e:
                raise RuntimeError(str(e))

            if 'apply' in fit_params:
                controller_options['apply'] = fit_params['apply']
                del controller_options['params']['apply']

                if 'model_name' not in controller_options and not fit_params['apply']:
                    raise RuntimeError('You must save a model if you are not applying it.')

            if 'partial_fit' in fit_params:
                partial_fit = fit_params['partial_fit']
                del controller_options['params']['partial_fit']

        if partial_fit:
            controller_options['processor'] = 'FitPartialProcessor'

        return controller_options, partial_fit

    def setup(self):
        """Get options, start controller and return command type.

        Returns:
            (dict): get info response (command type) and required fields
        """
        self.controller_options, self.partial_fit = self.handle_arguments(self.getinfo)
        self.controller = ChunkedController(self.getinfo, self.controller_options)
        required_fields = self.controller.get_required_fields()
        return {'type': CommandType.EVENTS, 'required_fields': required_fields}

    def get_output_body(self):
        """Collect output body from controller.

        Returns:
            (str): body
        """
        return self.controller.output_results()

    def handler(self, metadata, body):
        """Main handler we override from BaseChunkHandler.

        Args:
            metadata (dict): metadata information
            body (str): data payload from CEXC

        Returns:
            (dict): metadata to be sent back to CEXC
            output_body (str): data payload to be sent back to CEXC
        """
        if command_util.should_early_return(metadata):
            return {'type': CommandType.EVENTS}

        if command_util.is_getinfo_chunk(metadata):
            return self.setup()

        if self.getinfo.get('preview', False):
            logger.debug('Not running in preview.')
            return {'finished': True}

        finished_flag = metadata.get('finished', False)

        self.controller.load_data(body)

        # Partial fit should *always* execute on every chunk.
        # Non partial fit will execute on the last chunk.
        if self.partial_fit or finished_flag:
            self.controller.execute()
            output_body = self.get_output_body()
        else:
            output_body = None

        if finished_flag:
            self.controller.finalize()

        # Our final farewell
        self.log_performance_timers()
        return ({'finished': finished_flag}, output_body)

    def log_performance_timers(self):
        logger.debug(
            "read_time=%f, handle_time=%f, write_time=%f, csv_parse_time=%f, csv_render_time=%f"
            % (
                self._read_time,
                self._handle_time,
                self._write_time,
                self.controller._csv_parse_time,
                self.controller._csv_render_time,
            )
        )


if __name__ == "__main__":
    logger.debug("Starting forecast.py.")

    with Timer() as t:
        ForecastCommand(handler_data=BaseChunkHandler.DATA_RAW).run()

    log_uuid()
    log_fit_time(t.interval)
    logger.debug("Exiting gracefully. Byee!!")
