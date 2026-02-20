import os
import sys
import glob
import uuid
from typing import Callable, Optional, Dict, Any
from core.utils import *

# 全局进度存储（供 API 轮询）
_download_progress: Dict[str, Dict[str, Any]] = {}


def find_video_files(save_path='output'):
    """
    查找视频或音频媒体文件
    支持所有在config.yaml中定义的视频和音频格式
    """
    # 支持视频和音频文件
    all_formats = load_key("allowed_video_formats") + load_key("allowed_audio_formats")
    media_files = [file for file in glob.glob(save_path + "/*") if os.path.splitext(file)[1][1:].lower() in all_formats]
    # change \\ to /, this happen on windows
    if sys.platform.startswith('win'):
        media_files = [file.replace("\\", "/") for file in media_files]
    media_files = [file for file in media_files if not file.startswith("output/output")]
    if len(media_files) != 1:
        raise ValueError(f"Number of media files found {len(media_files)} is not unique. Please check.")
    return media_files[0]


def download_video_ytdlp(
    url: str,
    output_dir: str = "api/uploads",
    file_id: str = None,
    format_selector: str = None,
    progress_callback: Optional[Callable] = None
) -> dict:
    """
    使用 yt-dlp 下载 YouTube 视频

    Args:
        url: YouTube URL
        output_dir: 输出目录
        file_id: 唯一文件ID（用于文件名前缀，避免冲突）
        format_selector: 格式选择器（如 'bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best'）
        progress_callback: 进度回调函数

    Returns:
        {
            'filepath': '下载的文件路径',
            'title': '视频标题',
            'duration': 时长（秒）,
            'thumbnail': '缩略图URL',
            'uploader': '上传者名称'
        }
    """

    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp 未安装，请运行: uv add yt-dlp")

    # 生成唯一文件名前缀
    prefix = f"{file_id}_" if file_id else ""

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 如果没有指定格式选择器，使用默认（最佳 MP4）
    if not format_selector:
        format_selector = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best'

    # yt-dlp 配置（基于官方文档）
    ydl_opts = {
        # 使用指定的格式选择器
        'format': format_selector,

        # 文件名：视频标题（官方默认行为）
        # outtmpl 默认就是 '%(title)s.%(ext)s'，会自动使用视频标题
        'outtmpl': os.path.join(output_dir, f'{prefix}%(title)s.%(ext)s'),

        # 合并格式
        'merge_output_format': 'mp4',

        # 进度钩子（官方推荐方式）
        'progress_hooks': [lambda d: _progress_hook(d, file_id, progress_callback)],

        # 静默模式（避免 stdout 干扰）
        'quiet': True,
        'no_warnings': True,

        # 不下载缩略图和字幕
        'writethumbnail': False,
        'writesubtitles': False,
        'writeautomaticsub': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # 先获取视频信息（不下载）
        info = ydl.extract_info(url, download=False)

        # 使用 sanitize_info 确保可序列化（官方推荐）
        clean_info = ydl.sanitize_info(info)

        # 初始化进度
        _download_progress[file_id] = {
            'status': 'starting',
            'progress': 0,
            'total_bytes': clean_info.get('filesize', 0),
            'downloaded_bytes': 0,
            'message': '准备下载...',
            'title': clean_info.get('title', 'Unknown'),
        }

        # 开始下载
        ydl.download([url])

        # 查找下载的文件 - 遍历输出目录找到带 prefix 的视频文件
        final_filepath = None
        video_extensions = ['.mp4', '.webm', '.mkv']

        # 获取输出目录中所有文件
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                if filename.startswith(prefix) and any(filename.endswith(ext) for ext in video_extensions):
                    final_filepath = os.path.join(output_dir, filename)
                    break

        if not final_filepath or not os.path.exists(final_filepath):
            raise FileNotFoundError(f"下载完成但找不到文件: {prefix}*")

        # 更新进度为完成
        _download_progress[file_id] = {
            'status': 'completed',
            'progress': 100,
            'message': '下载完成',
            'filepath': final_filepath
        }

        return {
            'filepath': final_filepath,
            'title': clean_info.get('title', 'Unknown'),
            'duration': clean_info.get('duration', 0),
            'thumbnail': clean_info.get('thumbnail', ''),
            'uploader': clean_info.get('uploader', ''),
            'description': clean_info.get('description', ''),
        }


def _progress_hook(d: dict, file_id: str, callback: Optional[Callable] = None):
    """
    进度钩子函数（官方推荐的 progress_hooks 方式）

    d 字段包含：
    - status: 'downloading' | 'finished' | 'error'
    - downloaded_bytes: 已下载字节数
    - total_bytes: 总字节数
    - filename: 文件名
    - eta: 预计剩余时间（秒）
    - speed: 下载速度
    """

    if file_id not in _download_progress:
        return

    if d['status'] == 'downloading':
        total = d.get('total_bytes') or d.get('total_bytes_estimate') or d.get('filesize', 0)
        downloaded = d.get('downloaded_bytes', 0)

        if total > 0:
            progress = (downloaded / total) * 100
            speed = d.get('speed', 0)
            eta = d.get('eta', 0)

            # 获取文件名判断当前下载的是视频还是音频
            filename = d.get('filename', '')
            if 'video' in filename.lower() or filename.endswith('.webm') or filename.endswith('.mp4'):
                stage = '下载视频中'
            elif 'audio' in filename.lower() or filename.endswith('.m4a'):
                stage = '下载音频中'
            else:
                stage = '下载中'

            _download_progress[file_id] = {
                'status': 'downloading',
                'progress': round(progress, 1),
                'total_bytes': total,
                'downloaded_bytes': downloaded,
                'speed': speed,
                'eta': eta,
                'message': f'{stage}... {progress:.1f}%'
            }

    elif d['status'] == 'finished':
        # 可能是视频流完成，也可能是音频流完成
        # 检查是否有 postprocessor 在运行
        _download_progress[file_id] = {
            'status': 'post-processing',
            'progress': 95,
            'message': '正在合并视频和音频...'
        }

    elif d['status'] == 'error':
        _download_progress[file_id] = {
            'status': 'error',
            'progress': 0,
            'message': '下载失败'
        }

    # 调用自定义回调
    if callback:
        callback(d)


def get_download_progress(file_id: str) -> dict:
    """获取下载进度（供 API 轮询使用）"""
    return _download_progress.get(file_id, {'status': 'unknown', 'progress': 0, 'message': '未知状态'})


def parse_video_formats(url: str) -> dict:
    """
    解析 YouTube 视频的可用格式和质量选项

    Args:
        url: YouTube URL

    Returns:
        {
            'title': '视频标题',
            'thumbnail': '缩略图URL',
            'duration': 时长（秒）,
            'uploader': '上传者名称',
            'formats': [
                {
                    'format_id': '137+140',
                    'quality': '2K',
                    'ext': 'mp4',
                    'height': 1440,
                    'width': 2560,
                    'filesize': 450000000,  # 字节
                    'fps': 30,
                    'vcodec': 'avc1.640028',
                    'acodec': 'mp4a.40.2',
                    'format_selector': 'bestvideo[ext=mp4][height<=1440]+bestaudio/best[ext=mp4]',
                    'label': '1440p 2K (MP4) - 429 MB'
                },
                ...
            ]
        }
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp 未安装，请运行: uv add yt-dlp")

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # 获取视频信息
        info = ydl.extract_info(url, download=False)
        clean_info = ydl.sanitize_info(info)

        # 获取所有格式
        all_formats = clean_info.get('formats', [])

        # 分类格式：MP4 视频格式（带视频流）
        mp4_video_formats = [
            f for f in all_formats
            if f.get('ext') == 'mp4'
            and f.get('vcodec') != 'none'
            and f.get('height', 0) > 0
            and not f.get('only_video', False)  # 排除纯视频（需要音频）
        ]

        # 格式化输出
        format_options = []

        # 按分辨率分组（相同分辨率只保留最佳的）
        resolution_groups = {}
        for fmt in mp4_video_formats:
            height = fmt.get('height', 0)
            if height not in resolution_groups or fmt.get('filesize', 0) > resolution_groups[height].get('filesize', 0):
                resolution_groups[height] = fmt

        # 按分辨率排序（从高到低）
        sorted_heights = sorted(resolution_groups.keys(), reverse=True)

        for height in sorted_heights:
            # 跳过 480P 以下的低质量视频
            if height < 480:
                continue

            fmt = resolution_groups[height]

            # 获取质量标签
            quality_label = _get_quality_label(height)

            # 生成 format_selector（用于下载）
            # 对于 MP4 格式，使用 height 限制
            format_selector = f"bestvideo[ext=mp4][height<={height}]+bestaudio/best[ext=mp4]"

            format_options.append({
                'format_id': fmt.get('format_id', ''),
                'quality': quality_label,
                'ext': fmt.get('ext', 'mp4'),
                'height': height,
                'width': fmt.get('width', 0),
                'fps': fmt.get('fps', 0),
                'vcodec': fmt.get('vcodec', ''),
                'acodec': fmt.get('acodec', ''),
                'format_selector': format_selector,
                'label': f"{height}p"
            })

        # 添加"最佳"选项（推荐）
        if format_options:
            best_option = format_options[0].copy()
            best_option['label'] = "最佳"
            best_option['is_recommended'] = True
            format_options.insert(0, best_option)

        return {
            'title': clean_info.get('title', 'Unknown'),
            'thumbnail': clean_info.get('thumbnail', ''),
            'duration': clean_info.get('duration', 0),
            'uploader': clean_info.get('uploader', ''),
            'formats': format_options
        }


def _get_quality_label(height: int) -> str:
    """根据分辨率返回质量标签"""
    if height >= 4320:  # 8K
        return "8K"
    elif height >= 2880:  # 5K
        return "5K"
    elif height >= 2160:  # 4K
        return "4K"
    elif height >= 1440:  # 2K
        return "2K"
    elif height >= 1080:  # Full HD
        return "1080p Full HD"
    elif height >= 720:  # HD
        return "720p HD"
    elif height >= 480:  # SD
        return "480p SD"
    elif height >= 360:  #
        return "360p"
    elif height >= 240:
        return "240p"
    elif height >= 144:
        return "144p"
    else:
        return f"{height}p"


def _format_filesize(size_bytes: int) -> str:
    """格式化文件大小"""
    if not size_bytes:
        return "未知大小"

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            if unit == 'B':
                return f"{size_bytes} B"
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def clear_download_progress(file_id: str):
    """清理进度记录"""
    if file_id in _download_progress:
        del _download_progress[file_id]


if __name__ == '__main__':
    print(f"🎥 Found video file: {find_video_files()}")
