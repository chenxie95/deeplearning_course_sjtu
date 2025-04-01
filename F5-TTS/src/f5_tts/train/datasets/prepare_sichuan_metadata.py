import os
import argparse
from pathlib import Path
import pandas as pd

# 读取UTTRANSINFO.txt文件 数据格式如下
"""
CHANNEL	UTTRANS_ID	SPEAKER_ID	PROMPT	TRANSCRIPTION
C0	G0001_0009.wav	G0001		三国吴国顾雍是个什么样的人
C0	G0001_0010.wav	G0001		枣庄师范教学怎么样
C0	G0001_0011.wav	G0001		石家庄哪里有情侣专卖店
C0	G0001_0012.wav	G0001		天起越热，药味就弥漫得越快，这说明
C0	G0001_0013.wav	G0001		劲舞团如何卡歌
C0	G0001_0014.wav	G0001		多少女孩在恋爱期保持处女呀
"""

# 生成metadata.csv文件 数据格式如下
# 存储在跟UTTRANSINFO.txt同级目录下
"""
audio_file|text
WAV/G0001/G0001_0009.wav|三国吴国顾雍是个什么样的人
WAV/G0001/G0001_0010.wav|枣庄师范教学怎么样
WAV/G0001/G0001_0011.wav|石家庄哪里有情侣专卖店
WAV/G0001/G0001_0012.wav|天起越热，药味就弥漫得越快，这说明
WAV/G0001/G0001_0013.wav|劲舞团如何卡歌
WAV/G0001/G0001_0014.wav|多少女孩在恋爱期保持处女呀
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="UTTRANSINFO.txt 文件路径")
    return parser.parse_args()

def generate_metadata(uttransinfo_path, output_csv_path):
    """
    根据 UTTRANSINFO.txt 生成 metadata.csv。
    
    文件列顺序:
      0: CHANNEL
      1: UTTRANS_ID (音频文件名)
      2: SPEAKER_ID
      3: PROMPT
      4: TRANSCRIPTION (文本)
    """
    # 打开 UTTRANSINFO.txt 文件
    df = pd.read_csv(uttransinfo_path, sep="\t", header=0)
    df_filter = df[["UTTRANS_ID", "TRANSCRIPTION"]].copy()

    # 写入 metadata.csv，表头是 "audio_file|text"
    with output_csv_path.open('w', encoding='utf-8', newline='') as outfile:
        # 写表头 "audio_file|text"
        outfile.write("audio_file|text\n")
        # 写入数据
        for index, row in df_filter.iterrows():
            # 获取音频文件名和文本
            uttrans_id = row["UTTRANS_ID"].strip()        # G0001_0009.wav
            spk,_ = uttrans_id.split("_")
            uttrans_id = os.path.join("WAV", spk, uttrans_id)  # WAV/G0001/G0001_0009.wav
            transcription = row["TRANSCRIPTION"].strip()
            outfile.write(f"{uttrans_id}|{transcription}\n")

    print(f"[INFO] metadata.csv 文件已生成: {output_csv_path.resolve()}")

def main():
    args = parse_args()
    uttransinfo_path = Path(args.input)
    output_csv_path = uttransinfo_path.parent / "metadata.csv"
    generate_metadata(
        uttransinfo_path=args.input,
        output_csv_path=output_csv_path,
    )

if __name__ == "__main__":
    main()
