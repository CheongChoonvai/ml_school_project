"""
Logging configuration for OncoPredictAI

This module provides functions to configure logging.
"""

import os
import logging
import logging.config

def setup_logging(config):
    """
    Setup logging configuration.
    
    Args:
        config: Logging configuration dictionary
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join('outputs', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Get log level
    level = config.get('level', 'INFO')
    level = getattr(logging, level)
    
    # Get log format
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'app.log')),
            logging.StreamHandler()
        ]
    )
    
    # Log configuration complete
    logging.info('Logging configured')

def get_logger(name):
    """
    Get logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
