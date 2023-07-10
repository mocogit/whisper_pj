import argparse
import os
import pickle
import whisper
import time


def transcribe_files(prefix, model_names, use_existing_model=False):
    # 音声データが保存されているディレクトリ
    audio_dir = '../input/'

    for model_name in model_names:

        # モデルのロードまたはデシリアライズ
        model_path = f'whisper_model_{model_name}.pkl'
        if os.path.exists(model_path) and use_existing_model:
            print(f"Using existing model: {model_name}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            print(f"Downloading new model: {model_name}")
            model = whisper.load_model(model_name)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        start_time = time.time()  # 計測開始
        # 音声データのディレクトリ内の全ファイルを処理
        for audio_filename in os.listdir(audio_dir):
            # ファイル拡張子が.wav、.mp3、または.mp4でない場合、スキップ
            if not audio_filename.lower().endswith(('.wav', '.mp3', '.mp4', '.m4a')):
                continue

            audio_filepath = os.path.join(audio_dir, audio_filename)
            print(f"Processing file: {audio_filepath}")

            # 音声をテキストに変換
            transcription_data = model.transcribe(audio_filepath)
            transcription = transcription_data.get('text', '')  # キーが存在しない場合には空文字列を返す
            print(f"Transcription: {transcription}")  # デバッグ用のプリントを追加

            # ファイル名にプレフィックスを付けて保存
            output_filename = prefix + "_" + audio_filename.split(".")[0] + ".txt"
            output_dir = f'../output/transcriptions/{model_name}/'
            os.makedirs(output_dir, exist_ok=True)
            output_filepath = os.path.join(output_dir, output_filename)
            with open(output_filepath, "w") as file:
                file.write(str(transcription))  # 文字列化
            print(f"Saved transcription to: {output_filepath}")

        end_time = time.time()  # 計測終了
        elapsed_time = end_time - start_time  # 経過時間
        print(f"Time elapsed for {model_name}: {elapsed_time} sec")  # 結果の表示


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe audio files using the whisper model.')
    parser.add_argument('-p', '--prefix', help='The prefix to append to the output files.', required=True)
    parser.add_argument('-m', '--model_name', nargs='+', help='The whisper model names.', required=True)
    parser.add_argument('-u', '--use_existing_model', action='store_true',
                        help='Whether to use an existing model if available.')
    args = parser.parse_args()

    transcribe_files(args.prefix, args.model_name, args.use_existing_model)
