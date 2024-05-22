import os
import PIL
from imageio import imread
from moviepy.editor import ImageSequenceClip
import numpy as np
 
def make_video(image_folder, output_vid, fps=24):
    # 获取图片列表
    image_files = []
    # image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg", ".png"))]
    for idx in range(len(os.listdir(image_folder))):
        image_files.append(os.path.join(image_folder, f"{idx + 1}.png"))
        # print(imread(image_files[-1]).shape)
    #     image = PIL.Image.open(os.path.join(image_folder, f"{idx + 1}.png"))
    #     image = image.resize((1920, 1080))
    #     image.save(os.path.join("resize", f"{idx + 1}.png"))

    # for idx in range(len(os.listdir("resize"))):
    #     image_files.append(os.path.join("resize", f"{idx + 1}.png"))
    #     print(imread(image_files[-1]).shape)
        
    # 按文件名排序
    # image_files.sort()
    # print(image_files)
    # 创建视频剪辑
    clip = ImageSequenceClip(image_files, fps=fps)
    # 输出到文件
    clip.write_videofile(output_vid)

os.makedirs("output", exist_ok=True)
# 使用示例
for i in np.random.choice(range(1, 26), 4, replace=False):
    print(f"Making video {i}")
    make_video(f"pics/{i}", f"output/output_video_{i}.mp4", fps=24)