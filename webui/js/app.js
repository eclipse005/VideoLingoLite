/**
 * VideoLingoLite Web UI - 主应用入口
 * 重构后：文件和任务分离获取
 */

import { initToastAnimations } from './utils.js';
import { initFileUpload } from './components/fileUpload.js';
import { initFileList, updateBadges, renderFileList } from './components/fileList.js';
import { TermsManager } from './components/terms.js';
import { SettingsManager } from './components/settings.js';
import { setFiles, setTasks } from './state.js';

// ===== DOM 元素引用 =====
const elements = {
  termsBtn: null,
  settingsBtn: null
};

// 轮询定时器
let pollingTimer = null;
const POLL_INTERVAL = 2000;  // 2秒

// ===== 初始化 =====
document.addEventListener('DOMContentLoaded', () => {
  initElements();
  initEventListeners();
  initScrollAnimation();
  initApp();
});

function initElements() {
  elements.termsBtn = document.getElementById('termsBtn');
  elements.settingsBtn = document.getElementById('settingsBtn');
}

function initEventListeners() {
  // 导航按钮
  elements.termsBtn.addEventListener('click', () => TermsManager.open());
  elements.settingsBtn.addEventListener('click', () => SettingsManager.open());
}

function initApp() {
  // 初始化 Toast 动画
  initToastAnimations();

  // 初始化轮询（获取文件+任务状态）
  startPolling();

  // 初始化组件
  initFileUpload();
  initFileList();
  TermsManager.init();
  SettingsManager.init();

  // 更新徽章
  updateBadges();
}

// ===== 状态轮询 =====
function startPolling() {
  // 立即执行一次
  fetchState();

  // 定期轮询
  pollingTimer = setInterval(fetchState, POLL_INTERVAL);
}

/**
 * 获取完整状态（文件 + 任务）
 * 1. 获取文件列表（含 active_task_id）
 * 2. 收集所有 active_task_id，批量获取任务详情
 */
async function fetchState() {
  try {
    // 1. 获取文件列表
    const filesResponse = await fetch('/api/files');
    if (!filesResponse.ok) return;
    const files = await filesResponse.json();

    // 2. 收集所有活跃任务ID
    const activeTaskIds = files
      .map(f => f.active_task_id)
      .filter(id => id != null);

    // 3. 批量获取任务详情
    let tasks = {};
    if (activeTaskIds.length > 0) {
      const tasksResponse = await fetch('/api/tasks');
      if (tasksResponse.ok) {
        const tasksList = await tasksResponse.json();
        // 转换为 {task_id: TaskInfo} 格式
        tasks = Object.fromEntries(tasksList.map(t => [t.id, t]));
      }
    }

    // 4. 更新状态
    setFiles(files);
    setTasks(tasks);

    // 5. 触发渲染
    renderFileList();
  } catch (error) {
    console.error('获取状态失败:', error);
  }
}

/**
 * 导出 fetchState 供其他模块使用
 * 例如：上传文件、创建任务后需要刷新状态
 */
export { fetchState };

// ===== 滚动动画 =====
function initScrollAnimation() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('animated');
        }
      });
    },
    {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    }
  );

  document.querySelectorAll('.apple-animate-on-scroll').forEach((el) => {
    observer.observe(el);
  });
}
