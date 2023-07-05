import argparse
import os
import pickle
import whisper


def transcribe_files(prefix, model_name='large-v2', use_existing_model=False):
    model_path = f'whisper_model_{model_name}.pkl'
    if os.path.exists(model_path) and use_existing_model:
        print("Using existing model.")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        print("Downloading new model.")
        model = whisper.load_model(model_name)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    audio_dir = '../input/'

    for audio_filename in os.listdir(audio_dir):
        if not audio_filename.lower().endswith(('.wav', '.mp3', '.mp4')):
            continue

        audio_filepath = os.path.join(audio_dir, audio_filename)
        print(f"Processing file: {audio_filepath}")

        transcription_data = model.transcribe(audio_filepath)
        transcription = transcription_data.get('text', '')
        print(f"Transcription: {transcription}")

        output_filename = prefix + "_" + audio_filename.split(".")[0] + ".txt"
        output_dir = '../output/transcriptions/'
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, output_filename)
        with open(output_filepath, "w") as file:
            file.write(str(transcription))
        print(f"Saved transcription to: {output_filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe audio files using the whisper model.')
    parser.add_argument('-p', '--prefix', help='The prefix to append to the output files.', required=True)
    parser.add_argument('-m', '--model_name', help='The whisper model name.', default='large-v2')
    parser.add_argument('-u', '--use_existing_model', action='store_true',
                        help='Whether to use an existing model if available.')
    args = parser.parse_args()

    transcribe_files(args.prefix, args.model_name, args.use_existing_model)
