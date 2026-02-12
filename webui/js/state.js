/**
 * 状态管理模块
 * 重构后：文件和任务分离存储
 */

const state = {
  files: [],           // 文件列表（含 active_task_id）
  tasks: {},           // 任务字典 {task_id: TaskInfo}
  terms: [],
  isConfigured: false,
  confirmDialog: null
};

// 获取状态
export function getState() {
  return state;
}

// ===== 文件管理 =====
export function setFiles(files) {
  state.files = files;
}

export function addFiles(files) {
  state.files.push(...files);
}

export function removeFile(fileId) {
  state.files = state.files.filter(f => f.id !== fileId);
}

export function clearAllFiles() {
  state.files = [];
}

export function getFiles() {
  return state.files;
}

export function getFileById(fileId) {
  return state.files.find(f => f.id === fileId);
}

// ===== 任务管理 =====
export function setTasks(tasks) {
  state.tasks = tasks;
}

export function getTask(taskId) {
  return state.tasks[taskId] || null;
}

export function getFileTask(fileId) {
  const file = getFileById(fileId);
  if (!file || !file.active_task_id) return null;
  return getTask(file.active_task_id);
}

// ===== 其他状态 =====
export function setTerms(terms) {
  state.terms = terms;
}

export function getTerms() {
  return state.terms;
}

export function setConfigured(configured) {
  state.isConfigured = configured;
}

export function isConfigured() {
  return state.isConfigured;
}

export function showConfirmDialog(dialog) {
  state.confirmDialog = dialog;
}

export function hideConfirmDialog() {
  state.confirmDialog = null;
}

export function getConfirmDialog() {
  return state.confirmDialog;
}
