/**
 * 设置管理组件
 */

import { showToast, showProgressToast, apiGet, apiPut, apiPost, escapeHtml } from '../utils.js';

// API 渠道配置
const API_CHANNELS = {
  'api': '默认 API（通用）',
  'api_split': '分句 API',
  'api_summary': '摘要 API',
  'api_translate': '翻译 API',
  'api_reflection': '反思 API',
  'api_hotword': '热词矫正 API'
};

// 工具函数集合
const Utils = {
  escapeHtml,
  showToast
};

// ASR 运行时选项
const ASR_RUNTIMES = {
  'qwen': 'Qwen3-ASR（本地 GPU，52 种语言 + 22 种中文方言）',
  'custom': '自定义 API（用户配置端点）'
};

// Qwen3-ASR 模型选项
const QWEN_MODELS = {
  'Qwen3-ASR-0.6B': 'Qwen3-ASR-0.6B（速度快，精度较低）',
  'Qwen3-ASR-1.7B': 'Qwen3-ASR-1.7B（高精度，速度较慢）'
};

// 语言选项
const LANGUAGE_OPTIONS = {
  'en': '英语',
  'zh': '中文',
  'ja': '日语',
  'ko': '韩语',
  'es': '西班牙语',
  'fr': '法语',
  'de': '德语',
  'it': '意大利语',
  'ru': '俄语'
};

const SettingsManager = {
  // 状态
  data: {
    // 当前选中的标签页
    currentTab: 'api',

    // 当前选中的 API 渠道
    currentApiChannel: 'api',

    // API 渠道配置
    apiChannels: {
      'api': {
        key: 'DEEPSEEK_API_KEY',
        base_url: 'http://localhost:8088/v1',
        model: 'deepseek-chat'
      },
      'api_split': {
        key: 'DEEPSEEK_API_KEY',
        base_url: 'http://localhost:8088/v1',
        model: 'deepseek-chat'
      },
      'api_summary': {
        key: 'DEEPSEEK_API_KEY',
        base_url: 'http://localhost:8088/v1',
        model: 'deepseek-chat'
      },
      'api_translate': {
        key: 'DEEPSEEK_API_KEY',
        base_url: 'http://localhost:8088/v1',
        model: 'deepseek-chat'
      },
      'api_reflection': {
        key: 'DEEPSEEK_API_KEY',
        base_url: 'http://localhost:8088/v1',
        model: 'deepseek-chat'
      },
      'api_hotword': {
        key: 'DEEPSEEK_API_KEY',
        base_url: 'http://localhost:8088/v1',
        model: 'deepseek-chat'
      }
    },

    // ASR 设置
    asrLanguage: 'en',
    asrRuntime: 'qwen',
    asrModel: 'Qwen3-ASR-0.6B',

    // 目标语言
    targetLanguage: '简体中文',


    // 热词矫正（分组结构）
    hotwordCorrection: {
      enabled: false,
      activeGroupId: 'group-0',
      groups: [
        { id: 'group-0', name: '默认分组', keyterms: [] }
      ]
    },

    // 分组 UI 状态
    nextGroupIndex: 1,
    editingGroupId: null,
    editingGroupName: '',
    newGroupName: '',

    // 人声分离
    vocalSeparation: {
      enabled: false
    },

    // 高级设置展开状态
    advancedExpanded: false,

    // 应用到所有渠道
    applyToAll: false
  },

  // 初始化
  async init() {
    window.openSettingsManager = () => this.open();
    // 从后端加载配置
    await this.loadSettings();
  },

  // 从后端加载配置
  async loadSettings() {
    try {
      const config = await apiGet('/config');

      // 加载 API 渠道配置
      this.data.apiChannels = config.api_channels || {};

      // 加载 ASR 配置
      this.data.asrLanguage = config.asr?.language || 'en';
      this.data.asrRuntime = config.asr?.runtime || 'qwen';
      this.data.asrModel = config.asr?.model || 'Qwen3-ASR-0.6B';

      // 加载其他配置
      this.data.targetLanguage = config.target_language || '简体中文';

      // 加载热词矫正配置（分组结构）
      const hwGroups = config.hotword_correction?.groups || [];
      if (hwGroups.length > 0) {
        this.data.hotwordCorrection = {
          enabled: config.hotword_correction?.enabled || false,
          activeGroupId: config.hotword_correction?.active_group_id || hwGroups[0].id,
          groups: hwGroups
        };
        // 计算下一个分组索引
        const maxIndex = Math.max(...hwGroups.map(g => parseInt(g.id.replace('group-', '')) || 0));
        this.data.nextGroupIndex = maxIndex + 1;
      } else {
        // 如果没有分组，创建默认分组
        this.data.hotwordCorrection = {
          enabled: config.hotword_correction?.enabled || false,
          activeGroupId: 'group-0',
          groups: [{ id: 'group-0', name: '默认分组', keyterms: [] }]
        };
        this.data.nextGroupIndex = 1;
      }

      // 加载人声分离配置
      this.data.vocalSeparation = {
        enabled: config.vocal_separation?.enabled || false
      };

      // 加载高级配置
      this.data.advanced = config.advanced || {};

    } catch (error) {
      console.error('Failed to load settings:', error);
      showToast('配置加载失败，使用默认配置', 'error');
    }
  },

  // 打开设置模态框
  async open() {
    // 每次打开时都加载最新配置
    await this.loadSettings();
    
    const modalContainer = document.getElementById('modalContainer');
    modalContainer.innerHTML = this.render();
    document.body.style.overflow = 'hidden';

    // 渲染 API 配置
    this.renderApiConfig();

    // 渲染分组标签栏
    this.renderGroups();

    // 渲染热词列表
    this.renderHotwords();

    // 绑定事件
    this.bindEvents();
  },

  // 切换标签页
  switchTab(tabId) {
    if (!tabId) return;

    this.data.currentTab = tabId;

    // 更新标签按钮状态
    document.querySelectorAll('.settings-tab-btn').forEach(btn => {
      if (btn.dataset.tab === tabId) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    // 更新内容区域显示
    document.querySelectorAll('.settings-tab-content').forEach(content => {
      if (content.id === `tab-${tabId}`) {
        content.classList.add('active');
      } else {
        content.classList.remove('active');
      }
    });
  },

  // 关闭设置模态框
  close() {
    const modalContainer = document.getElementById('modalContainer');
    modalContainer.innerHTML = '';
    document.body.style.overflow = '';
  },

  // 渲染 API 配置表单
  renderApiConfig() {
    const config = this.data.apiChannels[this.data.currentApiChannel] || {
      key: '',
      base_url: '',
      model: ''
    };

    document.getElementById('apiKey').value = config.key || '';
    document.getElementById('apiBaseUrl').value = config.base_url || '';
    document.getElementById('apiModel').value = config.model || '';
  },

  // 切换 API 渠道
  switchApiChannel(channel) {
    this.data.currentApiChannel = channel;
    this.renderApiConfig();
  },

  // 绑定事件
  bindEvents() {
    // 标签页切换
    document.querySelectorAll('.settings-tab-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const tabId = e.currentTarget.dataset.tab;
        this.switchTab(tabId);
      });
    });

    // API 渠道选择
    const channelSelect = document.getElementById('apiChannelSelect');
    if (channelSelect) {
      channelSelect.addEventListener('change', (e) => {
        this.switchApiChannel(e.target.value);
      });
    }

    // ASR 运行时切换（显示/隐藏模型选择）
    const asrRuntimeSelect = document.getElementById('asrRuntime');
    if (asrRuntimeSelect) {
      asrRuntimeSelect.addEventListener('change', (e) => {
        const modelGroup = document.getElementById('asrModelGroup');
        if (modelGroup) {
          if (e.target.value === 'qwen') {
            modelGroup.style.display = '';
          } else {
            modelGroup.style.display = 'none';
          }
        }
      });
    }

    // 测试 API 按钮
    const testBtn = document.getElementById('testApiBtn');
    if (testBtn) {
      testBtn.addEventListener('click', () => this.testApiConnection());
    }

    // 应用到所有渠道复选框
    const applyToAllCheckbox = document.getElementById('applyToAll');
    if (applyToAllCheckbox) {
      applyToAllCheckbox.addEventListener('change', (e) => {
        this.data.applyToAll = e.target.checked;
      });
    }

    // 保存设置按钮
    const saveBtn = document.getElementById('saveSettingsBtn');
    if (saveBtn) {
      saveBtn.addEventListener('click', () => this.saveSettings());
    }

    // 高级设置展开/折叠
    const advancedToggle = document.getElementById('advancedToggle');
    if (advancedToggle) {
      advancedToggle.addEventListener('click', () => {
        this.data.advancedExpanded = !this.data.advancedExpanded;
        this.toggleAdvanced();
      });
    }

    // 添加术语按钮
    const addTermBtn = document.getElementById('addHotwordBtn');
    if (addTermBtn) {
      addTermBtn.addEventListener('click', () => this.addHotword());
    }

    // 点击外部取消选中热词标签
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.hotword-item')) {
        document.querySelectorAll('.hotword-item.selected').forEach(i => i.classList.remove('selected'));
      }
    });
  },

  // 测试 API 连接
  async testApiConnection() {
    const config = {
      channel: this.data.currentApiChannel,
      key: document.getElementById('apiKey').value,
      base_url: document.getElementById('apiBaseUrl').value,
      model: document.getElementById('apiModel').value
    };

    const toast = showProgressToast(`正在测试 ${API_CHANNELS[this.data.currentApiChannel]} 连接...`, 'info');

    try {
      const result = await apiPost('/test-connection', config);

      if (result.success) {
        toast.success(result.message);
      } else {
        toast.error(`连接失败: ${result.message}`);
      }
    } catch (error) {
      toast.error(`测试失败: ${error.message || '网络错误'}`);
    }
  },

  // 保存设置
  async saveSettings() {
    const toast = showProgressToast('正在保存设置...', 'info');

    try {
      // 先保存当前 API 渠道配置（修正顺序）
      if (this.data.applyToAll) {
        Object.keys(API_CHANNELS).forEach(channel => {
          this.data.apiChannels[channel] = {
            key: document.getElementById('apiKey')?.value || '',
            base_url: document.getElementById('apiBaseUrl')?.value || '',
            model: document.getElementById('apiModel')?.value || ''
          };
        });
      } else {
        this.data.apiChannels[this.data.currentApiChannel] = {
          key: document.getElementById('apiKey')?.value || '',
          base_url: document.getElementById('apiBaseUrl')?.value || '',
          model: document.getElementById('apiModel')?.value || ''
        };
      }

      // 收集所有设置
      const settings = {
        api_channels: this.data.apiChannels,
        asr: {
          language: document.getElementById('asrLanguage')?.value || this.data.asrLanguage,
          runtime: document.getElementById('asrRuntime')?.value || this.data.asrRuntime,
          model: document.getElementById('asrModel')?.value || this.data.asrModel
        },
        target_language: document.getElementById('targetLanguage')?.value || this.data.targetLanguage,
        hotword_correction: {
          enabled: document.getElementById('hotwordEnabled')?.checked ?? this.data.hotwordCorrection.enabled,
          active_group_id: this.data.hotwordCorrection.activeGroupId,
          groups: this.data.hotwordCorrection.groups
        },
        vocal_separation: {
          enabled: document.getElementById('vocalSeparation')?.checked ?? this.data.vocalSeparation.enabled
        },
        advanced: this.data.advanced
      };

      // 调用后端 API 保存配置
      await apiPut('/config', settings);

      toast.success('设置已保存');
    } catch (error) {
      console.error('Failed to save settings:', error);
      toast.error('保存失败，请重试');
    }
  },

  // 切换高级设置
  toggleAdvanced() {
    const advancedContent = document.getElementById('advancedContent');
    const advancedIcon = document.getElementById('advancedIcon');

    if (this.data.advancedExpanded) {
      advancedContent.classList.add('show');
      advancedIcon.style.transform = 'rotate(180deg)';
    } else {
      advancedContent.classList.remove('show');
      advancedIcon.style.transform = 'rotate(0deg)';
    }
  },

  // ==================== 分组管理方法 ====================

  // 切换激活分组
  activateGroup(groupId) {
    this.data.hotwordCorrection.activeGroupId = groupId;
    this.renderGroups();
    this.renderHotwords();
  },

  // 显示新建分组输入框
  showNewGroupInput() {
    this.data.newGroupName = '新建分组';
    this.renderGroups();
  },

  // 添加分组
  addGroup() {
    const name = this.data.newGroupName.trim();
    if (!name) {
      showToast('请输入分组名称', 'error');
      return;
    }

    const newGroup = {
      id: `group-${this.data.nextGroupIndex++}`,
      name: name,
      keyterms: []
    };

    this.data.hotwordCorrection.groups.push(newGroup);
    this.data.hotwordCorrection.activeGroupId = newGroup.id;
    this.data.newGroupName = '';
    this.renderGroups();
    this.renderHotwords();
  },

  // 删除分组
  deleteGroup(groupId) {
    if (this.data.hotwordCorrection.groups.length <= 1) {
      showToast('至少保留一个分组', 'error');
      return;
    }

    const idx = this.data.hotwordCorrection.groups.findIndex(g => g.id === groupId);
    this.data.hotwordCorrection.groups.splice(idx, 1);

    // 如果删除的是激活分组，激活第一个
    if (this.data.hotwordCorrection.activeGroupId === groupId) {
      this.data.hotwordCorrection.activeGroupId = this.data.hotwordCorrection.groups[0].id;
    }

    this.renderGroups();
    this.renderHotwords();
  },

  // 开始编辑分组名
  startEditGroup(groupId, currentName) {
    this.data.editingGroupId = groupId;
    this.data.editingGroupName = currentName;
    this.renderGroups();
  },

  // 保存编辑的分组名
  saveEditGroup() {
    if (!this.data.editingGroupId) return;

    const group = this.data.hotwordCorrection.groups.find(
      g => g.id === this.data.editingGroupId
    );
    if (group && this.data.editingGroupName.trim()) {
      group.name = this.data.editingGroupName.trim();
    }
    this.data.editingGroupId = null;
    this.data.editingGroupName = '';
    this.renderGroups();
  },

  // 渲染分组标签栏
  renderGroups() {
    const container = document.getElementById('groupsTabs');
    if (!container) return;

    const groups = this.data.hotwordCorrection.groups;
    const activeId = this.data.hotwordCorrection.activeGroupId;

    container.innerHTML = groups.map(group => {
      const isActive = group.id === activeId;
      const isEditing = this.data.editingGroupId === group.id;

      return `
        <div class="hotword-group-tab ${isActive ? 'active' : ''}"
             data-group-id="${group.id}"
             onclick="SettingsManager.activateGroup('${group.id}')">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
          </svg>
          ${isEditing ? `
            <input type="text" class="group-name-input"
                   value="${Utils.escapeHtml(this.data.editingGroupName)}"
                   oninput="SettingsManager.data.editingGroupName = this.value"
                   onkeydown="if(event.key==='Enter') SettingsManager.saveEditGroup()"
                   onblur="SettingsManager.saveEditGroup()"
                   onclick="event.stopPropagation()"
                   autofocus>
          ` : `
            <span class="group-name">${Utils.escapeHtml(group.name)}</span>
          `}
          <span class="group-count">(${group.keyterms.length})</span>
          <div class="group-actions" onclick="event.stopPropagation()">
            ${!isEditing ? `
              <button class="group-action-btn" onclick="SettingsManager.startEditGroup('${group.id}', '${Utils.escapeHtml(group.name).replace(/'/g, "\\'")}')">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                </svg>
              </button>
              ${groups.length > 1 ? `
                <button class="group-action-btn" onclick="SettingsManager.deleteGroup('${group.id}')">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <polyline points="3 6 5 6 21 6"/>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                  </svg>
                </button>
              ` : ''}
            ` : `
              <button class="group-action-btn" onclick="SettingsManager.saveEditGroup()">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <polyline points="20 6 9 17 4 12"/>
                </svg>
              </button>
            `}
          </div>
        </div>
      `;
    }).join('');

    // 新建分组输入框或按钮
    if (this.data.newGroupName) {
      container.innerHTML += `
        <div class="hotword-group-tab">
          <input type="text" class="group-name-input"
                 value="${Utils.escapeHtml(this.data.newGroupName)}"
                 oninput="SettingsManager.data.newGroupName = this.value"
                 onkeydown="if(event.key==='Enter') SettingsManager.addGroup()"
                 onblur="if(SettingsManager.data.newGroupName.trim()) SettingsManager.addGroup(); else SettingsManager.data.newGroupName=''; SettingsManager.renderGroups();"
                 onclick="event.stopPropagation()">
          <button class="group-action-btn" onclick="event.stopPropagation(); SettingsManager.addGroup()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
          </button>
        </div>
      `;
    } else {
      container.innerHTML += `
        <button class="hotword-group-tab hotword-group-add-btn" onclick="SettingsManager.showNewGroupInput()" title="新建分组">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="12" y1="5" x2="12" y2="19"/>
            <line x1="5" y1="12" x2="19" y2="12"/>
          </svg>
        </button>
      `;
    }
  },

  // ==================== 热词管理方法 ====================

  // 添加热词（支持英文逗号批量添加）
  addHotword() {
    const termInput = document.getElementById('newHotword');
    const input = termInput?.value.trim();

    if (!input) {
      showToast('请输入热词', 'error');
      return;
    }

    // 找到激活分组
    const activeGroup = this.data.hotwordCorrection.groups.find(
      g => g.id === this.data.hotwordCorrection.activeGroupId
    );

    if (!activeGroup) {
      showToast('未找到激活分组', 'error');
      return;
    }

    // 检测是否包含英文逗号（批量添加）
    if (input.includes(',')) {
      // 按英文逗号分割，去除每个热词首尾空格，过滤空字符串
      const terms = input
        .split(',')
        .map(t => t.trim())
        .filter(t => t.length > 0);

      if (terms.length === 0) {
        showToast('请输入有效的热词', 'error');
        return;
      }

      // 批量添加到激活分组
      activeGroup.keyterms.push(...terms);
      this.renderHotwords();

      if (termInput) {
        termInput.value = '';
      }

      showToast(`已添加 ${terms.length} 个热词`, 'success');
    } else {
      // 单个热词添加
      activeGroup.keyterms.push(input);
      this.renderHotwords();

      if (termInput) {
        termInput.value = '';
      }

      showToast('热词已添加', 'success');
    }
  },

  // 删除热词
  removeHotword(term) {
    const activeGroup = this.data.hotwordCorrection.groups.find(
      g => g.id === this.data.hotwordCorrection.activeGroupId
    );

    if (!activeGroup) {
      return;
    }

    const idx = activeGroup.keyterms.indexOf(term);
    if (idx > -1) {
      activeGroup.keyterms.splice(idx, 1);
      this.renderHotwords();
    }
  },

  // 渲染热词列表（当前激活分组的热词）
  renderHotwords() {
    const container = document.getElementById('hotwordsList');
    if (!container) return;

    // 找到激活分组
    const activeGroup = this.data.hotwordCorrection.groups.find(
      g => g.id === this.data.hotwordCorrection.activeGroupId
    );

    if (!activeGroup || activeGroup.keyterms.length === 0) {
      container.innerHTML = '';
      return;
    }

    container.innerHTML = activeGroup.keyterms.map((term) => {
      // 解析热词格式：如果有冒号，分成原文和译文
      if (term.includes(':')) {
        const [original, translation] = term.split(':');
        return `
          <div class="hotword-item">
            <span class="hotword-original">${Utils.escapeHtml(original.trim())}</span>
            <span class="hotword-arrow">→</span>
            <span class="hotword-translation">${Utils.escapeHtml(translation.trim())}</span>
            <button class="hotword-remove" onclick="SettingsManager.removeHotword('${Utils.escapeHtml(term).replace(/'/g, "\\'")}')">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>
          </div>
        `;
      } else {
        return `
          <div class="hotword-item">
            <span class="hotword-text">${Utils.escapeHtml(term)}</span>
            <button class="hotword-remove" onclick="SettingsManager.removeHotword('${Utils.escapeHtml(term).replace(/'/g, "\\'")}')">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>
          </div>
        `;
      }
    }).join('');
  },

  // 渲染热词列表 HTML（兼容方法，调用 renderHotwords）
  renderHotwordsList() {
    this.renderHotwords();
    return '';
  },

  // 渲染整个界面
  render() {
    return `
      <div class="modal-overlay">
        <div class="modal-content modal-content-settings">
          <button class="modal-close" onclick="SettingsManager.close()">&times;</button>

          <div class="settings-header">
            <h2 class="apple-heading-medium">设置</h2>
          </div>

          <!-- 标签页导航 -->
          <div class="settings-tabs-nav">
            <button class="settings-tab-btn ${this.data.currentTab === 'api' ? 'active' : ''}" data-tab="api">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                <path d="M2 17l10 5 10-5"/>
                <path d="M2 12l10 5 10-5"/>
              </svg>
              API 配置
            </button>
            <button class="settings-tab-btn ${this.data.currentTab === 'asr' ? 'active' : ''}" data-tab="asr">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                <line x1="12" y1="19" x2="12" y2="23"/>
                <line x1="8" y1="23" x2="16" y2="23"/>
              </svg>
              ASR 设置
            </button>
            <button class="settings-tab-btn ${this.data.currentTab === 'hotword' ? 'active' : ''}" data-tab="hotword">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="16 18 22 12 16 6"/>
                <polyline points="8 6 2 12 8 18"/>
              </svg>
              热词矫正
            </button>
            <button class="settings-tab-btn ${this.data.currentTab === 'advanced' ? 'active' : ''}" data-tab="advanced">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="3"/>
                <path d="M12 1v6m0 6v6"/>
                <path d="M5.6 5.6l4.2 4.2m4.4 4.4l4.2 4.2"/>
                <path d="M18.4 5.6l-4.2 4.2m-4.4 4.4l-4.2 4.2"/>
              </svg>
              高级设置
            </button>
          </div>

          <div class="settings-body">
            <!-- API 配置标签页 -->
            <div class="settings-tab-content ${this.data.currentTab === 'api' ? 'active' : ''}" id="tab-api">
              <div class="api-config">
                <div class="form-group">
                  <label>API 渠道</label>
                  <select id="apiChannelSelect" class="apple-select">
                    ${Object.entries(API_CHANNELS).map(([key, label]) => `
                      <option value="${key}" ${this.data.currentApiChannel === key ? 'selected' : ''}>${label}</option>
                    `).join('')}
                  </select>
                </div>

                <div class="api-config-form">
                  <div class="form-row">
                    <div class="form-group">
                      <label>API Key</label>
                      <input type="password" id="apiKey" class="apple-input" placeholder="输入 API Key 或环境变量名">
                      <small class="form-hint">支持环境变量，如 DEEPSEEK_API_KEY</small>
                    </div>

                    <div class="form-group">
                      <label>Base URL</label>
                      <input type="text" id="apiBaseUrl" class="apple-input" placeholder="http://localhost:8088/v1">
                    </div>
                  </div>

                  <div class="form-group">
                    <label>模型</label>
                    <input type="text" id="apiModel" class="apple-input" placeholder="deepseek-chat">
                  </div>

                  <div class="form-actions">
                    <button id="testApiBtn" class="apple-button apple-button-secondary">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                        <polyline points="22 4 12 14.01 9 11.01"/>
                      </svg>
                      测试连接
                    </button>

                    <label class="apply-all-checkbox">
                      <input type="checkbox" id="applyToAll">
                      <span>应用此配置到所有 API 渠道</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>

            <!-- ASR 设置标签页 -->
            <div class="settings-tab-content ${this.data.currentTab === 'asr' ? 'active' : ''}" id="tab-asr">
              <div class="settings-section">
                <h3 class="apple-heading-small">ASR 设置</h3>

                <div class="form-row">
                  <div class="form-group">
                    <label>源语言</label>
                    <select id="asrLanguage" class="apple-select">
                      ${Object.entries(LANGUAGE_OPTIONS).map(([key, label]) => `
                        <option value="${key}" ${this.data.asrLanguage === key ? 'selected' : ''}>${label}</option>
                      `).join('')}
                    </select>
                  </div>

                  <div class="form-group">
                    <label>目标语言</label>
                    <input type="text" id="targetLanguage" class="apple-input" value="${this.data.targetLanguage}" placeholder="简体中文">
                  </div>

                  <div class="form-group">
                    <label>ASR 引擎</label>
                    <select id="asrRuntime" class="apple-select">
                      ${Object.entries(ASR_RUNTIMES).map(([key, label]) => `
                        <option value="${key}" ${this.data.asrRuntime === key ? 'selected' : ''}>${label}</option>
                      `).join('')}
                    </select>
                  </div>

                  <div class="form-group" id="asrModelGroup" style="${this.data.asrRuntime === 'qwen' ? '' : 'display: none;'}">
                    <label>模型</label>
                    <select id="asrModel" class="apple-select">
                      ${Object.entries(QWEN_MODELS).map(([key, label]) => `
                        <option value="${key}" ${this.data.asrModel === key ? 'selected' : ''}>${label}</option>
                      `).join('')}
                    </select>
                  </div>
                </div>
              </div>

              <div class="settings-section">
                <div class="settings-toggles">
                  <label class="setting-toggle">
                    <input type="checkbox" id="vocalSeparation" ${this.data.vocalSeparation.enabled ? 'checked' : ''}>
                    <span class="toggle-label">
                      <span class="toggle-title">人声分离</span>
                      <span class="toggle-desc">嘈杂环境下自动分离人声，提升 ASR 准确率</span>
                    </span>
                    <span class="toggle-switch"></span>
                  </label>
                </div>
              </div>
            </div>

            <!-- 热词矫正标签页 -->
            <div class="settings-tab-content ${this.data.currentTab === 'hotword' ? 'active' : ''}" id="tab-hotword">
              <div class="settings-section">
                <div class="section-header">
                  <h3 class="apple-heading-small">热词矫正</h3>
                  <label class="switch">
                    <input type="checkbox" id="hotwordEnabled" ${this.data.hotwordCorrection.enabled ? 'checked' : ''}>
                    <span class="slider"></span>
                  </label>
                </div>

                <!-- 分组标签栏 -->
                <div class="hotword-groups-tabs" id="groupsTabs"></div>

                <!-- 当前分组热词列表 -->
                <div class="hotwords-config">
                  <div class="hotwords-list" id="hotwordsList"></div>

                  <!-- 添加热词输入框 -->
                  <div class="hotwords-add">
                    <input type="text" id="newHotword" class="apple-input" placeholder="单个添加：AI:Artificial Intelligence | 批量添加：API, SDK, CLI（英文逗号分隔）">
                    <button id="addHotwordBtn" class="apple-button apple-button-secondary">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="12" y1="5" x2="12" y2="19"/>
                        <line x1="5" y1="12" x2="19" y2="12"/>
                      </svg>
                      添加
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <!-- 高级设置标签页 -->
            <div class="settings-tab-content ${this.data.currentTab === 'advanced' ? 'active' : ''}" id="tab-advanced">
              <div class="settings-section">
                <h3 class="apple-heading-small">性能设置</h3>

                <div class="form-row">
                  <div class="form-group">
                    <label>LLM 并发数</label>
                    <input type="number" class="apple-input" value="16" min="1" max="32">
                    <small class="form-hint">本地 LLM 建议设置为 1</small>
                  </div>

                  <div class="form-group">
                    <label>摘要长度限制</label>
                    <input type="number" class="apple-input" value="8000" min="1000" max="32000" step="1000">
                    <small class="form-hint">本地 LLM 建议设置为 2000</small>
                  </div>
                </div>
              </div>

              <div class="settings-section">
                <h3 class="apple-heading-small">分句设置</h3>

                <div class="form-row">
                  <div class="form-group">
                    <label>暂停分割阈值（秒）</label>
                    <input type="number" class="apple-input" value="1.0" min="0" max="5" step="0.1">
                    <small class="form-hint">词间间隔超过此值时强制分句</small>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="settings-footer">
            <button id="saveSettingsBtn" class="apple-button">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"/>
              </svg>
              保存设置
            </button>
          </div>
        </div>
      </div>
    `;
  },

  // 渲染热词列表 HTML
  renderHotwordsList() {
    if (this.data.hotwordCorrection.terms.length === 0) {
      return '';
    }

    return this.data.hotwordCorrection.terms.map((term, index) => {
      // 解析热词格式：如果有冒号，分成原文和译文
      if (term.includes(':')) {
        const [original, translation] = term.split(':');
        return `
          <div class="hotword-item">
            <span class="hotword-original">${Utils.escapeHtml(original.trim())}</span>
            <span class="hotword-arrow">→</span>
            <span class="hotword-translation">${Utils.escapeHtml(translation.trim())}</span>
            <button class="hotword-remove" onclick="SettingsManager.removeHotword(${index})">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>
          </div>
        `;
      } else {
        return `
          <div class="hotword-item" data-index="${index}">
            <div class="hotword-content">
              <span class="hotword-text">${Utils.escapeHtml(term)}</span>
            </div>
            <div class="hotword-actions" onclick="event.stopPropagation(); SettingsManager.removeHotword(${index})">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </div>
          </div>
        `;
      }
    }).join('');
  }
};

// 导出到全局（用于 HTML 中的 onclick 事件）
window.SettingsManager = SettingsManager;

// ES6 导出
export { SettingsManager };
