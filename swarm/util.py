import inspect
from datetime import datetime
import tiktoken


def debug_print(debug: bool, *args: str) -> None:
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")


def merge_fields(target, source):
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: dict, delta: dict) -> None:
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """

    # if a partial fucntion is passed, get the original function

    print(f'FUNC: {func.__name__}')

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

def num_tokens_from_messages (messages, model="gpt-3.5-turbo-0613"):
        """Return the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        
        num_tokens = 0
        for message in messages:
            message = dict(message)
            print(message)
            num_tokens += len(encoding.encode(str(message['content'])))
            num_tokens += len(encoding.encode(message['role']))
            num_tokens += 12

        return num_tokens


def num_tokens_from_functions(functions, model="gpt-3.5-turbo-0613"):
        print(functions)
        """Return the number of tokens used by a list of functions."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        
        num_tokens = 0
        for function in functions:
            function = function['function']
            function_tokens = len(encoding.encode(function['name']))
            function_tokens += len(encoding.encode(function['description']))
            
            if 'parameters' in function:
                parameters = function['parameters']
                if 'properties' in parameters:
                    for propertiesKey in parameters['properties']:
                        function_tokens += len(encoding.encode(propertiesKey))
                        v = parameters['properties'][propertiesKey]
                        for field in v:
                            if field == 'type':
                                function_tokens += 2
                                function_tokens += len(encoding.encode(v['type']))
                            elif field == 'description':
                                function_tokens += 2
                                function_tokens += len(encoding.encode(v['description']))
                            elif field == 'enum':
                                function_tokens -= 3
                                for o in v['enum']:
                                    function_tokens += 3
                                    function_tokens += len(encoding.encode(o))
                            else:
                                print(f"Warning: not supported field {field}")
                    function_tokens += 11

            num_tokens += function_tokens

        num_tokens += 12 
        return num_tokens