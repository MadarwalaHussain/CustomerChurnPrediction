"""
Main utility functions for the Bank Churn prediction system.
Provides core file I/O operations for models, preprocessors, and data.
"""

import os
import sys
import dill
import yaml
import numpy as np
from typing import Any, Dict
from bank_churns.exception.exception import BankChurnException
from bank_churns.logging.logger import logging

def save_object(file_path:str, obj:Any)->None:
    """
    Save a Python object to disk using dill serialization.
    
    Why dill over pickle?
    - dill can serialize more complex objects (lambda functions, nested classes)
    - Better for ML pipelines with custom transformers
    - Compatible with sklearn pipelines and custom objects
    
    Args:
        file_path: Path where object should be saved (e.g., 'models/preprocessor.pkl')
        obj: Python object to serialize (model, preprocessor, transformer, etc.)
    
    Raises:
        BankChurnException: If serialization or file writing fails
    
    Example:
        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = StandardScaler()
        >>> save_object('artifacts/scaler.pkl', scaler)
    """
    try:
        logging.info(f'Saving object to: {file_path}')

        # create directory if it doesnt exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Serialize and save object
        with open(file_path,'wb') as f:
            dill.dump(obj, f)
        logging.info(f'object saved successfulyy to : {file_path}')
    
    except Exception as e:
        logging.error(f'Error saving object to {file_path}:{str(e)}')
        raise BankChurnException(e,sys)
    
def load_object(file_path:str)->Any:
    """
    Load a serialized Python object from disk.
    
    Args:
        file_path: Path to serialized object file
    
    Returns:
        Deserialized Python object (model, preprocessor, etc.)
    
    Raises:
        BankChurnException: If file doesn't exist or deserialization fails
    
    Example:
        >>> scaler = load_object('artifacts/scaler.pkl')
        >>> scaled_data = scaler.transform(data)
    """

    try:
        # check if file exists
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Load and seriralize object
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"object loaded succesfully from {file_path}")
        return obj 

    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {str(e)}")
        raise BankChurnException(e,sys)


def save_numpy_array(file_path:str, array:np.ndarray)-> None:
    """
    Save a NumPy array to disk in binary format.
    
    Why .npy format?
    - Fast I/O (binary format)
    - Preserves exact data types and shapes
    - Memory efficient for large arrays
    - Native NumPy format
    
    Args:
        file_path: Path where array should be saved (e.g., 'artifacts/train.npy')
        array: NumPy array to save (transformed features, predictions, etc.)
    
    Raises:
        BankChurnException: If saving fails
    
    Example:
        >>> import numpy as np
        >>> X_train_transformed = np.array([[1, 2, 3], [4, 5, 6]])
        >>> save_numpy_array('artifacts/train_transformed.npy', X_train_transformed)
    """
    try:
        logging.info(f'Saving numpy array to: {file_path}')

        # Create directory if not exits
        dir_path =os.path.dirname(file_path) 
        os.makedirs(dir_path,exist_ok=True)

        # save array in binary format
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
        logging.info(f"Numpy array saved successfully. Shape: {array.shape}")
    except Exception as e:
        logging.error(f"Error saving numpy array to {file_path}: {str(e)}")
        raise BankChurnException(e,sys)


def load_numpy_array(file_path:str)->np.ndarray:
    """
    Load a NumPy array from disk.
    
    Args:
        file_path: Path to .npy file
    
    Returns:
        Loaded NumPy array
    
    Raises:
        BankChurnException: If file doesn't exist or loading fails
    
    Example:
        >>> X_train = load_numpy_array('artifacts/train_transformed.npy')
        >>> print(X_train.shape)
    """
    try:
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        logging.info(f"Loading numpy array from: {file_path}")

        # Load array
        with open(file_path, 'rb') as file_obj:
            array=np.load(file_obj)
        logging.info(f"Numpy array loaded successfully. Shape: {array.shape}")
        return array
            
    except Exception as e:
        logging.error(f"Error loading numpy array from {file_path}: {str(e)}")
        raise BankChurnException(e,sys)
    

def write_yaml_file(file_path:str, content:Dict)->None:
    """
    Write a dictionary to a YAML file.
    
    Why YAML for reports?
    - Human-readable format
    - Easy to version control
    - Supports nested structures
    - Standard for configuration files
    
    Args:
        file_path: Path where YAML file should be saved
        content: Dictionary to write as YAML
    
    Raises:
        BankChurnException: If writing fails
    
    Example:
        >>> report = {
        ...     'validation_status': True,
        ...     'missing_columns': [],
        ...     'drift_detected': False
        ... }
        >>> write_yaml_file('artifacts/validation_report.yaml', report)
    """
    try:
        logging.info(f'Writing YAML file to {file_path}')
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Write YAML file
        with open(file_path,'w') as file_obj:
            yaml.dump(content,file_obj, default_flow_style=False, sort_keys=False)
        logging.info(f"YAML file written successfully to: {file_path}")

    except Exception as e:
        logging.error(f"Error writing YAML file to {file_path}: {str(e)}")
        raise BankChurnException(e,sys)


def read_yaml_file(file_path:str)->Dict:
    """
    Read a YAML file and return as dictionary.
    
    Args:
        file_path: Path to YAML file
    
    Returns:
        Dictionary containing YAML content
    
    Raises:
        BankChurnException: If file doesn't exist or reading fails
    
    Example:
        >>> config = read_yaml_file('config/schema.yaml')
        >>> print(config['numerical_columns'])
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        logging.info(f"Reading YAML file from: {file_path}")

        # Read YAML file
        with open(file_path, 'r') as file_obj:
            content = yaml.safe_load(file_obj)

        logging.info(f"YAML file read successfully from: {file_path}")
        return content

    except Exception as e:
        logging.error(f"Error reading YAML file from {file_path}: {str(e)}")
        raise BankChurnException(e, sys)


def get_file_size(file_path:str)->str:
    """
    Get human-readable file size.
    
    Args:
        file_path: Path to file
    
    Returns:
        File size as human-readable string (e.g., "1.5 MB")
    
    Example:
        >>> size = get_file_size('artifacts/model.pkl')
        >>> print(f"Model size: {size}")
    """
    try:
        if not os.path.exists(file_path):
            return 'File not found'
        size_bytes=os.path.getsize(file_path)

        # Convert to human readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes<1024:
                return f'{size_bytes:.2f} {unit}'
            size_bytes/=1024.0
        return f'{size_bytes:.2f} TB'
    except Exception as e:
        logging.error(f"Error getting file size for {file_path}: {str(e)}")
        return "Unknown"
