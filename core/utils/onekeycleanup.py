import os
import glob
from core._1_ytdlp import find_video_files
import shutil

def cleanup(history_dir="history", task_id=None):
    # Get video file name
    video_file = find_video_files()
    video_name = video_file.split("/")[1]
    video_name = os.path.splitext(video_name)[0]
    video_name = sanitize_filename(video_name)

    # Create required folders
    os.makedirs(history_dir, exist_ok=True)

    # Use task_id in directory name if provided
    if task_id:
        video_history_dir = os.path.join(history_dir, f"{video_name}_{task_id}")
    else:
        video_history_dir = os.path.join(history_dir, video_name)
    log_dir = os.path.join(video_history_dir, "log")
    gpt_log_dir = os.path.join(video_history_dir, "gpt_log")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(gpt_log_dir, exist_ok=True)

    moved_count = 0

    # Move non-log files
    for file in glob.glob("output/*"):
        if not file.endswith(('log', 'gpt_log')):
            if move_file(file, video_history_dir):
                moved_count += 1

    # Move log files
    for file in glob.glob("output/log/*"):
        if move_file(file, log_dir):
            moved_count += 1

    # Move gpt_log files
    for file in glob.glob("output/gpt_log/*"):
        if move_file(file, gpt_log_dir):
            moved_count += 1

    # Delete empty output directories
    try:
        os.rmdir("output/log")
        os.rmdir("output/gpt_log")
        os.rmdir("output")
    except OSError:
        pass  # Ignore errors when deleting directories

    print(f"✅ 归档完成: {moved_count} 个文件已移动到 {video_history_dir}")

def move_file(src, dst):
    """静默移动文件，返回是否成功"""
    try:
        # Get the source file name
        src_filename = os.path.basename(src)
        # Use os.path.join to ensure correct path and include file name
        dst = os.path.join(dst, sanitize_filename(src_filename))

        if os.path.exists(dst):
            if os.path.isdir(dst):
                # If destination is a folder, try to delete its contents
                shutil.rmtree(dst, ignore_errors=True)
            else:
                # If destination is a file, try to delete it
                os.remove(dst)

        shutil.move(src, dst, copy_function=shutil.copy2)
        return True
    except PermissionError:
        try:
            shutil.copy2(src, dst)
            os.remove(src)
            return True
        except Exception:
            return False
    except Exception:
        return False

def sanitize_filename(filename):
    # Remove or replace disallowed characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

if __name__ == "__main__":
    cleanup()