<template>
  <el-container class="layout">
    <!-- ============ 左侧边栏 ============ -->
    <el-aside width="260px" class="sidebar">
      <div class="brand">🏥 智慧问诊Agent系统</div>

      <el-button type="primary" class="new-chat" @click="newConversation">
        <el-icon><Plus /></el-icon>&nbsp;开启新对话
      </el-button>

      <!-- 对话历史 -->
      <div class="section">
        <div class="section-title">对话历史</div>
        <div class="conv-list">
          <div
            v-for="c in conversations" :key="c.id"
            class="conv-item" :class="{ active: c.id === sessionId }"
            @click="openConversation(c.id)"
          >
            <span class="conv-title">{{ c.title }}</span>
            <el-icon class="conv-del" @click.stop="removeConversation(c.id)"><Delete /></el-icon>
          </div>
          <div v-if="!conversations.length" class="empty-tip">暂无历史对话</div>
        </div>
      </div>

      <!-- 我能帮您 -->
      <div class="section">
        <div class="section-title">我能帮您</div>
        <div class="func-item">🏥 看病挂号：症状推荐科室</div>
        <div class="func-item">💊 治疗建议：用药信息查询</div>
        <div class="func-item">📚 医学知识：健康疑问解答</div>
      </div>

      <!-- 系统状态 -->
      <div class="section">
        <div class="section-title">系统状态</div>
        <div class="status-row"><span class="dot" :class="statusNeo4j ? 'ok' : 'bad'"></span>知识图谱</div>
        <div class="status-row"><span class="dot" :class="statusVector ? 'ok' : 'bad'"></span>向量检索</div>
        <div class="status-row"><span class="dot" :class="statusAgent ? 'ok' : 'bad'"></span>Agent 系统</div>
      </div>
    </el-aside>

    <!-- ============ 右侧主区 ============ -->
    <el-container>
      <el-header height="auto" style="padding:0">
        <DisclaimerBanner />
      </el-header>

      <el-main class="chat-main" ref="chatMainRef">
        <!-- 欢迎页 -->
        <div v-if="!messages.length" class="welcome">
          <h2>您好，我是智慧问诊助手 👋</h2>
          <p>根据症状推荐就诊科室、提供用药信息与医学知识解答</p>
          <div class="examples">
            <div v-for="q in examples" :key="q" class="example" @click="input = q; send()">{{ q }}</div>
          </div>
        </div>

        <!-- 消息列表 -->
        <ChatMessage
          v-for="(m, i) in messages" :key="i"
          :message="m" :loading="loading && i === messages.length - 1"
        />
      </el-main>

      <el-footer height="auto" class="input-area">
        <div class="input-row">
          <el-input
            v-model="input" placeholder="请描述您的症状或健康问题…（Enter 发送）"
            size="large" :disabled="loading" @keyup.enter="send" clearable
          />
          <el-button type="primary" size="large" :loading="loading" :disabled="loading" @click="send">
            {{ loading ? '思考中' : '发送' }}
          </el-button>
        </div>
      </el-footer>
    </el-container>
  </el-container>
</template>

<script setup>
import { nextTick, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import ChatMessage from '../components/ChatMessage.vue'
import DisclaimerBanner from '../components/DisclaimerBanner.vue'
import {
  consult, consultStream, deleteConversation, getConversation,
  getConversations, health
} from '../api/index.js'

const STORAGE_KEY = 'smc_session_id'

const messages = ref([])
const input = ref('')
const loading = ref(false)
const sessionId = ref('')
const conversations = ref([])
const statusNeo4j = ref(false)
const statusVector = ref(false)
const statusAgent = ref(false)
const chatMainRef = ref(null)

const examples = [
  '我最近头痛、头晕，应该挂什么科？',
  '高血压吃什么药好？',
  '什么是糖尿病？',
  '我突然胸痛、呼吸困难怎么办？'
]

function newSessionId() {
  return crypto.randomUUID()
}

function scrollToBottom() {
  nextTick(() => {
    const el = chatMainRef.value?.$el
    if (el) el.scrollTop = el.scrollHeight
  })
}

async function refreshStatus() {
  try {
    const h = await health()
    statusNeo4j.value = !!h.neo4j_connected
    statusVector.value = !!h.vector_index_loaded
    statusAgent.value = !!h.agent_system_ready
  } catch { /* 服务未启动时静默 */ }
}

async function refreshConversations() {
  try {
    conversations.value = await getConversations()
  } catch { /* 服务未启动时静默 */ }
}

function welcomeMessage() {
  messages.value = []
}

function newConversation() {
  sessionId.value = newSessionId()
  localStorage.setItem(STORAGE_KEY, sessionId.value)
  welcomeMessage()
}

function toMsg(row) {
  return {
    role: row.role === 'user' ? 'user' : 'ai',
    content: row.content,
    intent: row.intent || '',
    symptoms: [], departments: [], medications: [], disclaimers: [], warnings: [],
    linked_entities: [], progressSteps: [], durationMs: 0
  }
}

async function openConversation(id) {
  try {
    const rows = await getConversation(id)
    sessionId.value = id
    localStorage.setItem(STORAGE_KEY, id)
    messages.value = rows.map(toMsg)
    scrollToBottom()
  } catch {
    ElMessage.error('恢复对话失败（对话可能已删除）')
  }
}

async function removeConversation(id) {
  try {
    await ElMessageBox.confirm('确定删除该对话吗？', '提示', { type: 'warning' })
  } catch { return }
  try {
    await deleteConversation(id)
    if (id === sessionId.value) newConversation()
    await refreshConversations()
    ElMessage.success('已删除')
  } catch {
    ElMessage.error('删除失败')
  }
}

/** 发送：默认走 SSE 流式（progress 帧实时渲染节点进度），失败自动降级非流式 */
async function send() {
  const query = input.value.trim()
  if (!query || loading.value) return
  input.value = ''
  loading.value = true

  messages.value.push({ role: 'user', content: query })
  messages.value.push({
    role: 'ai', content: '', intent: '', symptoms: [], departments: [],
    medications: [], disclaimers: [], warnings: [], linked_entities: [],
    progressSteps: [], durationMs: 0
  })
  // 关键：从数组取响应式代理。若闭包直接持有字面量对象，
  // onAnswer 等回调的赋值会绕过 Vue 响应式，答案将永远不渲染。
  const aiMsg = messages.value[messages.value.length - 1]
  scrollToBottom()

  try {
    await consultStream(query, sessionId.value, {
      onProgress: (p) => {
        aiMsg.progressSteps.push(p)
        scrollToBottom()
      },
      onAnswer: (resp) => {
        aiMsg.content = resp.answer
        aiMsg.intent = resp.intent
        aiMsg.symptoms = resp.symptoms || []
        aiMsg.departments = resp.departments || []
        aiMsg.medications = resp.medications || []
        aiMsg.disclaimers = resp.disclaimers || []
        aiMsg.warnings = resp.warnings || []
        aiMsg.linked_entities = resp.linked_entities || []
        aiMsg.durationMs = resp.duration_ms
      },
      onError: (e) => {
        aiMsg.content = `抱歉，处理出现问题：${e.message}`
      }
    })
    if (!aiMsg.content) {
      // 流结束但没有 answer 帧 → 降级非流式
      await fallbackConsult(query, aiMsg)
    }
  } catch {
    // 连接失败自动降级回非流式（对用户透明）
    await fallbackConsult(query, aiMsg)
  } finally {
    loading.value = false
    scrollToBottom()
    refreshConversations()
  }
}

async function fallbackConsult(query, aiMsg) {
  try {
    const resp = await consult(query, sessionId.value)
    Object.assign(aiMsg, {
      content: resp.answer, intent: resp.intent,
      symptoms: resp.symptoms || [], departments: resp.departments || [],
      medications: resp.medications || [], disclaimers: resp.disclaimers || [],
      warnings: resp.warnings || [], linked_entities: resp.linked_entities || [],
      durationMs: resp.duration_ms
    })
  } catch (err) {
    aiMsg.content = err.code === 'ECONNABORTED'
      ? '请求超时，请稍后重试或简化您的问题。'
      : '服务暂时不可用，请确认后端已启动（python scripts/start_api.py）。'
  }
}

onMounted(async () => {
  sessionId.value = localStorage.getItem(STORAGE_KEY) || newSessionId()
  localStorage.setItem(STORAGE_KEY, sessionId.value)
  await Promise.all([refreshStatus(), refreshConversations()])
  // 刷新页面恢复当前对话（列表为空或 id 失效则回欢迎页）
  if (conversations.value.some((c) => c.id === sessionId.value)) {
    await openConversation(sessionId.value)
  }
})
</script>

<style scoped>
.layout { height: 100%; }

.sidebar {
  background: #0f2b1d; color: #e8f5e9; display: flex; flex-direction: column;
  padding: 16px 12px; overflow-y: auto;
}
.brand { font-size: 16px; font-weight: 600; padding: 4px 8px 16px; }
.new-chat { width: 100%; margin-bottom: 16px; }

.section { margin-bottom: 18px; }
.section-title { font-size: 12px; color: #8db6a0; margin: 0 8px 8px; }
.conv-list { display: flex; flex-direction: column; gap: 2px; }
.conv-item {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 10px; border-radius: 6px; cursor: pointer; font-size: 13px;
}
.conv-item:hover { background: rgba(255, 255, 255, 0.08); }
.conv-item.active { background: rgba(103, 194, 58, 0.25); }
.conv-title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.conv-del { opacity: 0; flex-shrink: 0; }
.conv-item:hover .conv-del { opacity: 0.8; }
.empty-tip { font-size: 12px; color: #5f8570; padding: 4px 10px; }

.func-item { font-size: 13px; color: #c8e6c9; padding: 4px 8px; }
.status-row { font-size: 13px; padding: 3px 8px; display: flex; align-items: center; gap: 8px; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot.ok { background: #67c23a; }
.dot.bad { background: #f56c6c; }

.chat-main { background: #f6f8f7; overflow-y: auto; padding: 20px 8%; }
.welcome { text-align: center; margin-top: 12vh; }
.welcome h2 { color: #303133; margin-bottom: 8px; }
.welcome p { color: #909399; margin-bottom: 24px; }
.examples { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }
.example {
  background: #fff; border: 1px solid #e4e7ed; border-radius: 8px;
  padding: 10px 16px; font-size: 13px; cursor: pointer; color: #606266;
}
.example:hover { border-color: #67c23a; color: #67c23a; }

.input-area { padding: 12px 8% 18px; background: #f6f8f7; }
.input-row { display: flex; gap: 10px; }
</style>
