/**
 * 术语管理组件
 */

import { escapeHtml, showToast, apiGet, apiPut, apiPatch, apiDelete } from '../utils.js';

// 创建 Utils 命名空间
const Utils = {
  escapeHtml,
  showToast
};

const TermsManager = {
  // 状态
  data: {
    terms: [],
    editingIndex: null,
    showImport: false,
    searchTerm: '',
    showClearConfirm: false
  },

  // 初始化
  async init() {
    window.openTermsManager = () => this.open();
    // 从后端加载术语
    await this.loadTerms();
  },

  // 从后端加载术语
  async loadTerms() {
    try {
      const result = await apiGet('/terms');
      this.data.terms = result.terms || [];
      this.updateBadges();
    } catch (error) {
      console.error('Failed to load terms:', error);
      // 如果是第一次运行，后端会创建空文件，忽略错误
      if (error.message.includes('404')) {
        this.data.terms = [];
      }
    }
  },

  // 保存术语到后端
  async saveTerms() {
    try {
      await apiPut('/terms', { terms: this.data.terms });
      this.updateBadges();
    } catch (error) {
      console.error('Failed to save terms:', error);
      Utils.showToast('保存失败，请重试', 'error');
      throw error;
    }
  },

  // 打开术语管理模态框
  open() {
    const modalContainer = document.getElementById('modalContainer');
    modalContainer.innerHTML = this.render();
    document.body.style.overflow = 'hidden';

    // 渲染术语列表
    this.renderTermsList();

    // 绑定事件
    this.bindEvents();
  },

  // 关闭术语管理模态框
  close() {
    const modalContainer = document.getElementById('modalContainer');
    modalContainer.innerHTML = '';
    document.body.style.overflow = '';
  },

  // 绑定事件
  bindEvents() {
    // 搜索输入
    const searchInput = document.getElementById('terms-search');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        this.data.searchTerm = e.target.value;
        this.renderTermsList();
      });
    }

    // 添加术语按钮
    const addBtn = document.getElementById('terms-add-btn');
    if (addBtn) {
      addBtn.addEventListener('click', () => this.addTerm());
    }

    // 导入/导出/清空按钮
    const importBtn = document.getElementById('terms-import-btn');
    if (importBtn) {
      importBtn.addEventListener('click', () => this.toggleImport());
    }

    const exportBtn = document.getElementById('terms-export-btn');
    if (exportBtn) {
      exportBtn.addEventListener('click', () => this.exportTerms());
    }

    const clearBtn = document.getElementById('terms-clear-btn');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => this.showClearConfirm());
    }

    // 导入确认按钮
    const importConfirmBtn = document.getElementById('terms-import-confirm');
    if (importConfirmBtn) {
      importConfirmBtn.addEventListener('click', () => this.importTerms());
    }

    // 导入取消按钮
    const importCancelBtn = document.getElementById('terms-import-cancel');
    if (importCancelBtn) {
      importCancelBtn.addEventListener('click', () => this.toggleImport());
    }

    // 点击外部取消选中术语标签
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.term-item')) {
        document.querySelectorAll('.term-item.selected').forEach(i => i.classList.remove('selected'));
      }
    });
  },

  // 添加术语
  addTerm() {
    const originalInput = document.getElementById('terms-original');
    const translationInput = document.getElementById('terms-translation');
    const notesInput = document.getElementById('terms-notes');

    const original = originalInput.value.trim();
    const translation = translationInput.value.trim();
    const notes = notesInput.value.trim();

    if (!original || !translation) {
      Utils.showToast('请输入原文和译文', 'error');
      return;
    }

    this.data.terms.push({ original, translation, notes: notes || '' });
    this.saveTerms();

    // 清空输入框
    originalInput.value = '';
    translationInput.value = '';
    notesInput.value = '';

    // 更新术语数量显示
    const termsCount = document.querySelector('.terms-count');
    if (termsCount) {
      termsCount.textContent = `${this.data.terms.length} 个术语`;
    }

    // 更新按钮状态
    const exportBtn = document.getElementById('terms-export-btn');
    const clearBtn = document.getElementById('terms-clear-btn');
    if (exportBtn) exportBtn.disabled = false;
    if (clearBtn) clearBtn.disabled = false;

    // 重新渲染术语列表
    this.renderTermsList();
    Utils.showToast('术语添加成功', 'success');
  },

  // 删除术语
  async removeTerm(index) {
    try {
      // 调用后端 API 删除术语
      await apiDelete(`/terms/${index}`);

      // 更新前端状态
      this.data.terms.splice(index, 1);
      this.updateBadges();

      // 更新术语数量显示
      const termsCount = document.querySelector('.terms-count');
      if (termsCount) {
        termsCount.textContent = `${this.data.terms.length} 个术语`;
      }

      // 更新按钮状态
      const exportBtn = document.getElementById('terms-export-btn');
      const clearBtn = document.getElementById('terms-clear-btn');
      if (exportBtn) exportBtn.disabled = this.data.terms.length === 0;
      if (clearBtn) clearBtn.disabled = this.data.terms.length === 0;

      this.renderTermsList();
      Utils.showToast('术语已删除', 'success');
    } catch (error) {
      console.error('Failed to delete term:', error);
      Utils.showToast('删除失败，请重试', 'error');
    }
  },

  // 开始编辑术语
  startEdit(index) {
    this.data.editingIndex = index;
    this.renderTermsList();
  },

  // 取消编辑
  cancelEdit() {
    this.data.editingIndex = null;
    this.renderTermsList();
  },

  // 保存编辑
  async saveEdit(index) {
    const originalInput = document.getElementById(`terms-edit-original-${index}`);
    const translationInput = document.getElementById(`terms-edit-translation-${index}`);
    const notesInput = document.getElementById(`terms-edit-notes-${index}`);

    const original = originalInput.value.trim();
    const translation = translationInput.value.trim();
    const notes = notesInput.value.trim();

    if (!original || !translation) {
      Utils.showToast('请输入原文和译文', 'error');
      return;
    }

    try {
      // 使用 PATCH 更新单个术语
      await apiPatch(`/terms/${index}`, { original, translation, notes: notes || '' });

      // 更新本地数据
      this.data.terms[index] = { original, translation, notes: notes || '' };
      this.data.editingIndex = null;

      // 重新渲染列表
      this.renderTermsList();
      this.updateBadges();

      Utils.showToast('术语已更新', 'success');
    } catch (error) {
      console.error('Failed to update term:', error);
      Utils.showToast('更新失败，请重试', 'error');
    }
  },

  // 切换导入面板
  toggleImport() {
    this.data.showImport = !this.data.showImport;

    // 重新渲染整个界面（因为导入面板状态改变了）
    const modalContainer = document.getElementById('modalContainer');
    modalContainer.innerHTML = this.render();
    this.renderTermsList();
    this.bindEvents();
  },

  // 导入术语
  importTerms() {
    const textarea = document.getElementById('terms-import-text');
    const text = textarea.value.trim();

    if (!text) {
      Utils.showToast('请输入要导入的术语', 'error');
      return;
    }

    // 解析文本：每行一个，原文:译文
    const lines = text.split('\n');
    let importCount = 0;

    lines.forEach(line => {
      line = line.trim();
      if (!line) return;

      const parts = line.split(':');
      if (parts.length >= 2) {
        const original = parts[0].trim();
        const translation = parts.slice(1).join(':').trim();
        const notes = '';

        this.data.terms.push({ original, translation, notes });
        importCount++;
      }
    });

    this.saveTerms();
    this.data.showImport = false;

    // 更新术语数量显示
    const termsCount = document.querySelector('.terms-count');
    if (termsCount) {
      termsCount.textContent = `${this.data.terms.length} 个术语`;
    }

    // 重新渲染整个界面（关闭导入面板）
    const modalContainer = document.getElementById('modalContainer');
    modalContainer.innerHTML = this.render();
    this.renderTermsList();
    this.bindEvents();

    Utils.showToast(`成功导入 ${importCount} 个术语`, 'success');
  },

  // 导出术语
  exportTerms() {
    if (this.data.terms.length === 0) {
      Utils.showToast('没有可导出的术语', 'error');
      return;
    }

    const content = this.data.terms
      .map(term => `${term.original}:${term.translation}`)
      .join('\n');

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'terms.txt';
    a.click();
    URL.revokeObjectURL(url);

    Utils.showToast('术语导出成功', 'success');
  },

  // 显示清空确认
  showClearConfirm() {
    if (this.data.terms.length === 0) {
      Utils.showToast('没有可清空的术语', 'error');
      return;
    }

    this.data.showClearConfirm = true;

    // 重新渲染整个界面（显示确认对话框）
    const modalContainer = document.getElementById('modalContainer');
    modalContainer.innerHTML = this.render();
    this.renderTermsList();
    this.bindEvents();
  },

  // 确认清空
  confirmClear() {
    this.data.terms = [];
    this.saveTerms();
    this.data.showClearConfirm = false;

    // 更新术语数量显示
    const termsCount = document.querySelector('.terms-count');
    if (termsCount) {
      termsCount.textContent = `${this.data.terms.length} 个术语`;
    }

    // 重新渲染整个界面（关闭确认对话框）
    const modalContainer = document.getElementById('modalContainer');
    modalContainer.innerHTML = this.render();
    this.renderTermsList();
    this.bindEvents();

    Utils.showToast('已清空所有术语', 'success');
  },

  // 取消清空
  cancelClear() {
    this.data.showClearConfirm = false;

    // 重新渲染整个界面（关闭确认对话框）
    const modalContainer = document.getElementById('modalContainer');
    modalContainer.innerHTML = this.render();
    this.renderTermsList();
    this.bindEvents();
  },

  // 获取过滤后的术语
  getFilteredTerms() {
    if (!this.data.searchTerm.trim()) {
      return this.data.terms;
    }

    const searchLower = this.data.searchTerm.toLowerCase();
    return this.data.terms.filter(term =>
      term.original.toLowerCase().includes(searchLower) ||
      term.translation.toLowerCase().includes(searchLower)
    );
  },

  // 更新徽章
  updateBadges() {
    const termsBadge = document.getElementById('termsBadge');
    if (!termsBadge) return;

    if (this.data.terms.length > 0) {
      termsBadge.textContent = this.data.terms.length;
      termsBadge.classList.add('show');
    } else {
      termsBadge.classList.remove('show');
    }
  },

  // 渲染术语列表
  renderTermsList() {
    const filteredTerms = this.getFilteredTerms();
    const termsListContainer = document.getElementById('terms-list');

    if (filteredTerms.length === 0) {
      termsListContainer.innerHTML = `
        <div class="terms-empty">
          ${this.data.searchTerm ? '没有找到匹配的术语' : '暂无术语，请添加术语或导入术语列表'}
        </div>
      `;
      return;
    }

    termsListContainer.innerHTML = filteredTerms.map((term, index) => {
      const actualIndex = this.data.terms.indexOf(term);

      if (this.data.editingIndex === actualIndex) {
        // 编辑模式
        return `
          <div class="term-item term-item-editing">
            <div class="term-edit-form">
              <div class="term-edit-row">
                <div class="term-edit-field">
                  <label>原文</label>
                  <input type="text" id="terms-edit-original-${actualIndex}" value="${Utils.escapeHtml(term.original)}" placeholder="原文">
                </div>
                <div class="term-edit-field">
                  <label>译文</label>
                  <input type="text" id="terms-edit-translation-${actualIndex}" value="${Utils.escapeHtml(term.translation)}" placeholder="译文">
                </div>
                <div class="term-edit-field">
                  <label>说明</label>
                  <input type="text" id="terms-edit-notes-${actualIndex}" value="${Utils.escapeHtml(term.notes)}" placeholder="说明（可选）">
                </div>
              </div>
              <div class="term-edit-actions">
                <button class="apple-button apple-button-emerald" onclick="TermsManager.saveEdit(${actualIndex})">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
                    <polyline points="17 21 17 13 7 13 7 21 17 21"/>
                  </svg>
                  保存
                </button>
                <button class="apple-button apple-button-ghost apple-button-red" onclick="TermsManager.cancelEdit()">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                  取消
                </button>
              </div>
            </div>
          </div>
        `;
      } else {
        // 显示模式 - 标签式
        return `
          <div class="term-item" data-index="${actualIndex}">
            <div class="term-tag-content">
              <span class="term-original">${Utils.escapeHtml(term.original)}</span>
              <span class="term-arrow">→</span>
              <span class="term-translation">${Utils.escapeHtml(term.translation)}</span>
            </div>
            <div class="term-actions" onclick="event.stopPropagation();">
              <span class="term-action-btn" onclick="TermsManager.startEdit(${actualIndex})" title="编辑">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5a2.121 2.121 0 0 1 3-3z"/>
                </svg>
              </span>
              <span class="term-action-btn term-action-delete" onclick="TermsManager.removeTerm(${actualIndex})" title="删除">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                  <line x1="18" y1="6" x2="6" y2="18"/>
                  <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </span>
            </div>
          </div>
        `;
      }
    }).join('');

    // 添加点击事件监听
    termsListContainer.querySelectorAll('.term-item').forEach(item => {
      item.addEventListener('click', (e) => {
        // 移除其他选中状态
        termsListContainer.querySelectorAll('.term-item.selected').forEach(i => i.classList.remove('selected'));
        // 切换当前选中状态
        item.classList.toggle('selected');
      });
    });
  },

  // 渲染整个界面
  render() {
    const filteredTerms = this.getFilteredTerms();

    return `
      <div class="modal-overlay">
        <div class="modal-content modal-content-terms">
          <button class="modal-close" onclick="TermsManager.close()">&times;</button>

          <div class="terms-header">
            <div class="terms-title">
              <h2 class="apple-heading-medium">术语管理</h2>
              <span class="terms-count">${this.data.terms.length} 个术语</span>
            </div>
          </div>

          <div class="terms-body">
            <!-- 添加术语 -->
            <div class="terms-section">
              <h3 class="apple-heading-small">添加新术语</h3>
              <div class="terms-add-form">
                <input type="text" id="terms-original" class="terms-input" placeholder="原文" onkeypress="if(event.key==='Enter')TermsManager.addTerm()">
                <input type="text" id="terms-translation" class="terms-input" placeholder="译文" onkeypress="if(event.key==='Enter')TermsManager.addTerm()">
                <input type="text" id="terms-notes" class="terms-input terms-input-notes" placeholder="说明（可选）" onkeypress="if(event.key==='Enter')TermsManager.addTerm()">
                <button id="terms-add-btn" class="apple-button apple-button-emerald">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="12" y1="5" x2="12" y2="19"/>
                    <line x1="5" y1="12" x2="19" y2="12"/>
                  </svg>
                  添加
                </button>
              </div>
            </div>

            <!-- 导入/导出 -->
            <div class="terms-section">
              <h3 class="apple-heading-small">导入/导出</h3>
              <div class="terms-actions">
                <button id="terms-import-btn" class="apple-button apple-button-secondary">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                  </svg>
                  导入术语
                </button>
                <button id="terms-export-btn" class="apple-button apple-button-secondary" ${this.data.terms.length === 0 ? 'disabled' : ''}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7 10 12 15 17 10"/>
                    <line x1="12" y1="15" x2="12" y2="3"/>
                  </svg>
                  导出术语
                </button>
                <button id="terms-clear-btn" class="apple-button apple-button-ghost apple-button-red" ${this.data.terms.length === 0 ? 'disabled' : ''}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="3 6 5 6 21 6"/>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                  </svg>
                  清空全部
                </button>
              </div>

              <!-- 导入文本框 -->
              ${this.data.showImport ? `
                <div class="terms-import-panel">
                  <textarea id="terms-import-text" class="terms-import-textarea" placeholder="请输入术语，每行一个，原文和译文用冒号(:)分隔，例如：原文:译文..."></textarea>
                  <div class="terms-import-actions">
                    <button id="terms-import-confirm" class="apple-button apple-button-secondary">确认导入</button>
                    <button id="terms-import-cancel" class="apple-button apple-button-ghost">取消</button>
                  </div>
                </div>
              ` : ''}
            </div>

            <!-- 术语列表 -->
            <div class="terms-section">
              <div class="terms-list-header">
                <h3 class="apple-heading-small">术语列表</h3>
                <div class="terms-search">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="11" cy="11" r="8"/>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                  </svg>
                  <input type="text" id="terms-search" class="terms-search-input" placeholder="搜索..." value="${Utils.escapeHtml(this.data.searchTerm)}">
                </div>
              </div>
              <div id="terms-list" class="terms-list">
                <!-- 术语列表将在这里渲染 -->
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 清空确认对话框 -->
      ${this.data.showClearConfirm ? `
        <div class="confirm-dialog-overlay" onclick="TermsManager.cancelClear()">
          <div class="confirm-dialog-content" onclick="event.stopPropagation()">
            <div class="confirm-dialog-header">
              <h3 class="apple-heading-small">确认清空</h3>
              <button class="confirm-dialog-close" onclick="TermsManager.cancelClear()">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <line x1="18" y1="6" x2="6" y2="18"/>
                  <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </button>
            </div>
            <div class="confirm-dialog-body">
              <p class="confirm-dialog-message">确定要清空所有 ${this.data.terms.length} 个术语吗？此操作不可恢复。</p>
            </div>
            <div class="confirm-dialog-footer">
              <button class="apple-button apple-button-ghost" onclick="TermsManager.cancelClear()">取消</button>
              <button class="apple-button apple-button-danger" onclick="TermsManager.confirmClear()">确认清空</button>
            </div>
          </div>
        </div>
      ` : ''}
    `;
  }
};

// ES6 导出
export { TermsManager };

// 导出到全局（用于 HTML 中的 onclick 事件）
window.TermsManager = TermsManager;
