# whisper_pj

## Code Example:
```
$ python main.py -p your_prefix -m large-v2 -u
```
The output files are saved in the ../output/transcriptions/model_name/ directory.

## Argument details:

> -p or --prefix: The prefix to append to the output files. This argument is required.
> 
> -m or --model_name: The name of the Whisper model　to use. This argument is required.
> 
> -u or --use_existing_model: If this option is specified, the program will use an existing model if one is available, and download a new model if not. This argument is optional, and if not specified, the program will always download a new model.
