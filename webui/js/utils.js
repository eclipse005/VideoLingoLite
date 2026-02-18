/**
 * 工具函数模块
 */

// 生成唯一 ID
export function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// HTML 转义
export function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// 文件大小格式化
export function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

// 获取文件类型（音频/视频）
export function getFileType(fileName) {
  const ext = fileName.toLowerCase().split('.').pop();
  const typeMap = {
    'mp3': 'audio',
    'wav': 'audio',
    'm4a': 'audio',
    'ogg': 'audio',
    'flac': 'audio',
    'mp4': 'video',
    'webm': 'video',
    'mkv': 'video',
    'avi': 'video',
    'mov': 'video'
  };
  return typeMap[ext] || 'unknown';
}

// 验证文件类型
export function isValidFile(file) {
  const validExtensions = ['.mp3', '.mp4', '.wav', '.m4a', '.avi', '.mov', '.mkv', '.ogg', '.flac', '.webm'];
  const fileName = file.name.toLowerCase();
  return validExtensions.some(ext => fileName.endsWith(ext));
}

// 获取文件图标
export function getFileIcon(type) {
  const icons = {
    audio: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="16" y1="8" x2="16" y2="14"/><line x1="8" y1="8" x2="8" y2="14"/></svg>',
    video: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3" fill="currentColor" opacity="0.3"/><polygon points="5 3 19 12 5 21 5 3"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>',
    unknown: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>'
  };
  return icons[type] || icons.unknown;
}

// Toast 消息管理
let activeToasts = [];
let toastIdCounter = 0;

// Toast 图标
const TOAST_ICONS = {
  success: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`,
  error: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`,
  info: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`
};

// 显示 Toast 消息
export function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  const toastId = ++toastIdCounter;

  toast.className = `toast toast-${type}`;
  toast.dataset.id = toastId;

  const icon = TOAST_ICONS[type] || TOAST_ICONS.info;
  const escapedMessage = escapeHtml(message);

  toast.innerHTML = `
    <div style="display: flex; align-items: center; gap: 12px;">
      ${icon}
      <span>${escapedMessage}</span>
    </div>
  `;

  // 计算位置（堆叠在最上方）
  const topOffset = 80 + (activeToasts.length * 70);
  toast.style.cssText = `
    position: fixed;
    top: ${topOffset}px;
    right: 24px;
    padding: 12px 20px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    font-size: 14px;
    z-index: 10001;
    animation: slideIn 0.3s ease;
  `;

  document.body.appendChild(toast);
  activeToasts.push({ id: toastId, element: toast });

  // 更新所有 toast 的位置
  updateToastPositions();

  setTimeout(() => {
    toast.style.animation = 'slideOut 0.3s ease';
    setTimeout(() => {
      toast.remove();
      activeToasts = activeToasts.filter(t => t.id !== toastId);
      updateToastPositions();
    }, 300);
  }, 3000);
}

// 显示带进度的 Toast（返回更新函数）
export function showProgressToast(message, type = 'info') {
  const toast = document.createElement('div');
  const toastId = ++toastIdCounter;

  toast.className = `toast toast-${type}`;
  toast.dataset.id = toastId;

  const iconContainerId = `toast-icon-${toastId}`;
  const messageSpanId = `toast-msg-${toastId}`;
  const progressBarId = `toast-progress-${toastId}`;

  toast.innerHTML = `
    <div style="display: flex; flex-direction: column; gap: 8px; width: auto;">
      <div style="display: flex; align-items: center; gap: 12px;">
        <span id="${iconContainerId}">${TOAST_ICONS[type] || TOAST_ICONS.info}</span>
        <span id="${messageSpanId}" style="flex: 1;">${escapeHtml(message)}</span>
      </div>
      <div id="${progressBarId}" style="display: none;">
        <div style="height: 4px; background: #f3f4f6; border-radius: 2px; overflow: hidden;">
          <div class="progress-bar-fill" style="height: 100%; background: #3b82f6; border-radius: 2px; transition: width 0.2s ease; width: 0%;"></div>
        </div>
      </div>
    </div>
  `;

  const topOffset = 80 + (activeToasts.length * 70);
  toast.style.cssText = `
    position: fixed;
    top: ${topOffset}px;
    right: 24px;
    padding: 12px 20px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    font-size: 14px;
    z-index: 10001;
    animation: slideIn 0.3s ease;
  `;

  document.body.appendChild(toast);
  activeToasts.push({ id: toastId, element: toast, isProgress: true });
  updateToastPositions();

  // 定时器（用于完成后自动关闭）
  let autoCloseTimer = null;

  // 返回更新和控制函数
  const progressControl = {
    update: (newMessage, newType = null, percent = null) => {
      const msgSpan = document.getElementById(messageSpanId);
      if (msgSpan) msgSpan.textContent = newMessage;

      // 更新图标
      if (newType) {
        const iconContainer = document.getElementById(iconContainerId);
        if (iconContainer) {
          iconContainer.innerHTML = TOAST_ICONS[newType] || TOAST_ICONS.info;
        }
        // 更新 className
        toast.className = `toast toast-${newType}`;
      }

      // 更新进度条
      if (percent !== null) {
        const progressBar = document.getElementById(progressBarId);
        const barFill = progressBar?.querySelector('.progress-bar-fill');
        if (progressBar) progressBar.style.display = 'block';
        if (barFill) barFill.style.width = `${percent}%`;
      } else {
        // 隐藏进度条
        const progressBar = document.getElementById(progressBarId);
        if (progressBar) progressBar.style.display = 'none';
      }
    },
    success: (message = '完成') => {
      // 变成成功状态，3秒后自动消失
      progressControl.update(message, 'success', null);

      // 清除之前的定时器
      if (autoCloseTimer) clearTimeout(autoCloseTimer);

      autoCloseTimer = setTimeout(() => {
        progressControl.close();
      }, 3000);
    },
    error: (message = '失败') => {
      // 变成错误状态，3秒后自动消失
      progressControl.update(message, 'error', null);

      if (autoCloseTimer) clearTimeout(autoCloseTimer);

      autoCloseTimer = setTimeout(() => {
        progressControl.close();
      }, 3000);
    },
    close: () => {
      if (autoCloseTimer) clearTimeout(autoCloseTimer);
      toast.style.animation = 'slideOut 0.3s ease';
      setTimeout(() => {
        toast.remove();
        activeToasts = activeToasts.filter(t => t.id !== toastId);
        updateToastPositions();
      }, 300);
    }
  };

  return progressControl;
}

// 更新所有 toast 的位置
function updateToastPositions() {
  activeToasts.forEach((toast, index) => {
    const topOffset = 80 + (index * 70);
    toast.element.style.top = `${topOffset}px`;
  });
}

// 添加 Toast 动画样式
export function initToastAnimations() {
  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideIn {
      from {
        opacity: 0;
        transform: translateX(100%);
      }
      to {
        opacity: 1;
        transform: translateX(0);
      }
    }
    @keyframes slideOut {
      from {
        opacity: 1;
        transform: translateX(0);
      }
      to {
        opacity: 0;
        transform: translateX(100%);
      }
    }
  `;
  document.head.appendChild(style);
}

// ============ API 调用函数 ============

// API 基础 URL
const API_BASE_URL = '/api';

// 通用 GET 请求
export async function apiGet(endpoint) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`API GET ${endpoint} failed:`, error);
    throw error;
  }
}

// 通用 POST 请求
export async function apiPost(endpoint, data) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`API POST ${endpoint} failed:`, error);
    throw error;
  }
}

// 通用 PUT 请求
export async function apiPut(endpoint, data) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`API PUT ${endpoint} failed:`, error);
    throw error;
  }
}

// 通用 PATCH 请求
export async function apiPatch(endpoint, data) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`API PATCH ${endpoint} failed:`, error);
    throw error;
  }
}

// 通用 DELETE 请求
export async function apiDelete(endpoint) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'DELETE'
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`API DELETE ${endpoint} failed:`, error);
    throw error;
  }
}
