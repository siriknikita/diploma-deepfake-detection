from typing import Any, cast

import numpy as np
from numpy._typing import NDArray
from pydantic import BaseModel, GetCoreSchemaHandler, PositiveFloat, PositiveInt
from pydantic_core import core_schema


# The core concept: create a class that Pydantic can understand.
# This class, NumpyArray, will inherit from np.ndarray.
class NumpyArray(NDArray[np.float64]):
    """
    A custom Pydantic-compatible class for NumPy arrays.

    This class enables Pydantic to validate and serialize NumPy ndarray objects
    by implementing the `__get_pydantic_core_schema__` method.
    """

    # This class method is the key to Pydantic's custom type handling.
    # Pydantic calls this method to get the validation and serialization rules
    # for the `NumpyArray` type.
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Generates the Pydantic core schema for the `NumpyArray` type.

        Args:
            source_type (Any): The type that the schema is being generated for.
            handler (GetCoreSchemaHandler): A handler for generating schemas for other types.

        Returns:
            A `core_schema.CoreSchema` object that Pydantic can use.
        """

        # 1. Define the validation logic.
        # This function converts an input (like a list of lists) into a NumPy array.
        def validate_from_list(value: list[Any]) -> NDArray[np.float64]:
            """
            Converts a list of lists to a NumPy array.

            Args:
                value (list[Any]): A list of lists representing the array data.

            Returns:
                A NumPy array constructed from the input list.
            """
            return np.array(value)

        # 2. Define the serialization logic.
        # This function converts a NumPy array back into a standard Python list.
        def serialize_to_list(
            value: NDArray[np.float64],
        ) -> float | list[float] | list[list[float]] | list[list[list[float]]]:
            """
            Converts a NumPy array to a list for serialization.

            Args:
                value (NDArray[np.float64]): A NumPy array to be serialized.

            Returns:
                A list representation of the NumPy array.
            """
            # Explicitly cast the return value to resolve the type checker error
            return cast(
                float | list[float] | list[list[float]] | list[list[list[float]]],
                value.tolist(),
            )

        # 3. Build the Pydantic Core Schema.
        # We use `core_schema.no_info_after_validator_function` to specify that
        # our custom validator function (`validate_from_list`) should be run
        # after Pydantic has validated that the input is a list.
        return core_schema.no_info_after_validator_function(
            validate_from_list,
            # The schema for the input value. We expect a list of any type.
            core_schema.list_schema(core_schema.any_schema()),
            # Add the serialization schema to handle the conversion back to a list.
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_to_list
            ),
        )


class ProjectSettings(BaseModel):
    """
    Project settings model. Defines configuration parameters for the project.

    Attributes:
        `window_size` (PositiveInt): The size of the window, must be a positive integer.
        `padding` (PositiveInt | PositiveFloat): The padding value, can be a positive integer or float.
    """

    window_size: PositiveInt
    padding: PositiveInt | PositiveFloat
