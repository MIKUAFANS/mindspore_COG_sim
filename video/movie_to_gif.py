from moviepy.editor import VideoFileClip

clip = (VideoFileClip("output/output_video_1.mp4"))
clip.write_gif("../assets/show.gif")
