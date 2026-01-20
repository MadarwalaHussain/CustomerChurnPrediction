"""
Custom exception handler for the Bank Churn prediction system.
Provides detailed error tracking with file, line number, and traceback.
"""

import sys
from typing import Optional


class BankChurnException(Exception):
    """
    Custom exception class for bank churn prediction system.
    Captures detailed error information including file and line number.
    """

    def __init__(self, error_message: Exception, error_details: sys):
        """
        Initialize custom exception with detailed error tracking.
        
        Args:
            error_message: The original exception
            error_details: sys module for extracting traceback
        """
        super().__init__(error_message)
        self.error_message = self._get_detailed_error_message(
            error_message=error_message,
            error_details=error_details
        )

    @staticmethod
    def _get_detailed_error_message(error_message: Exception, error_details: sys) -> str:
        """
        Extract detailed error information from exception.
        
        Args:
            error_message: Original exception
            error_details: sys module for traceback
            
        Returns:
            Formatted error message with file and line number
        """
        _, _, exc_tb = error_details.exc_info()

        # Extract file name and line number from traceback
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown"
        line_number = exc_tb.tb_lineno if exc_tb else 0

        error_message_detail = (
            f"Error occurred in script: [{file_name}] "
            f"at line number: [{line_number}] "
            f"with error message: [{str(error_message)}]"
        )

        return error_message_detail

    def __str__(self) -> str:
        """String representation of the exception."""
        return self.error_message

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"BankChurnException({self.error_message})"
