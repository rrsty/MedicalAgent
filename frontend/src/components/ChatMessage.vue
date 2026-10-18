<template>
  <div class="msg" :class="message.role === 'user' ? 'msg-user' : 'msg-ai'">
    <!-- 急症警告：置于该条消息最上方醒目展示 -->
    <el-alert
      v-for="(w, i) in message.warnings || []"
      :key="'w' + i"
      type="error"
      :title="w"
      :closable="false"
      show-icon
      class="warning-alert"
    />

    <!-- Agent 执行进度轨迹（SSE progress 帧） -->
    <div v-if="message.role === 'ai' && message.progressSteps && message.progressSteps.length && loading"
         class="progress-steps">
      <span v-for="(s, i) in message.progressSteps" :key="i" class="step">
        <el-icon><Loading /></el-icon> {{ s.label }}
        <span v-if="s.facts && s.facts.length" class="step-facts">{{ s.facts.join(' · ') }}</span>
      </span>
    </div>

    <!-- 正文（Markdown 渲染，DOMPurify 消毒防 XSS） -->
    <div class="bubble" v-html="rendered"></div>

    <!-- 结构化标签 -->
    <div v-if="message.symptoms && message.symptoms.length" class="tags">
      <el-tag v-for="s in message.symptoms" :key="'s' + s" type="info" size="small">🩺 {{ s }}</el-tag>
    </div>
    <div v-if="message.departments && message.departments.length" class="tags">
      <el-tag v-for="d in message.departments" :key="'d' + d" type="success" size="small">🏥 {{ d }}</el-tag>
    </div>
    <div v-if="message.medications && message.medications.length" class="tags">
      <el-tag v-for="m in message.medications" :key="'m' + m.name" type="warning" size="small">💊 {{ m.name }}</el-tag>
    </div>

    <!-- 知识定位（实体链接） -->
    <div v-if="message.linked_entities && message.linked_entities.length" class="tags">
      <el-tag
        v-for="(le, i) in message.linked_entities" :key="'le' + i"
        size="small" effect="plain" class="link-tag"
      >🔍 {{ le.input_entity }} → {{ le.matched_entity }}（{{ le.similarity }}）</el-tag>
    </div>

    <!-- 免责声明块 -->
    <div v-if="message.disclaimers && message.disclaimers.length" class="disclaimer">
      <div v-for="(d, i) in message.disclaimers" :key="'dc' + i">{{ d }}</div>
    </div>

    <!-- 元信息 -->
    <div v-if="message.role === 'ai' && (message.durationMs || message.intent)" class="meta">
      耗时 {{ message.durationMs }}ms · 意图 {{ message.intent }}
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

const props = defineProps({
  message: { type: Object, required: true },
  loading: { type: Boolean, default: false }
})

const rendered = computed(() => {
  const raw = marked.parse(props.message.content || '', { breaks: true })
  return DOMPurify.sanitize(raw)
})
</script>

<style scoped>
.msg { display: flex; flex-direction: column; margin: 10px 0; }
.msg-user { align-items: flex-end; }
.msg-ai { align-items: flex-start; }

.bubble {
  max-width: 78%;
  padding: 10px 14px;
  border-radius: 10px;
  font-size: 14px;
  line-height: 1.7;
  word-break: break-word;
}
.msg-user .bubble { background: #409eff; color: #fff; border-bottom-right-radius: 2px; white-space: pre-wrap; }
.msg-ai .bubble { background: #f4f4f5; color: #303133; border-bottom-left-radius: 2px; }
.msg-ai .bubble :deep(h1),
.msg-ai .bubble :deep(h2),
.msg-ai .bubble :deep(h3) { margin: 8px 0 4px; font-size: 15px; }
.msg-ai .bubble :deep(p) { margin: 4px 0; }
.msg-ai .bubble :deep(ul), .msg-ai .bubble :deep(ol) { padding-left: 20px; margin: 4px 0; }
.msg-ai .bubble :deep(code) { background: #e9e9eb; padding: 1px 4px; border-radius: 3px; }
.msg-ai .bubble :deep(table) { border-collapse: collapse; margin: 6px 0; }
.msg-ai .bubble :deep(th), .msg-ai .bubble :deep(td) { border: 1px solid #dcdfe6; padding: 4px 8px; }

.warning-alert { max-width: 78%; margin-bottom: 6px; }

.progress-steps { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px; max-width: 78%; }
.step {
  display: inline-flex; align-items: center; gap: 4px;
  font-size: 12px; color: #67c23a; background: #f0f9eb;
  border: 1px solid #e1f3d8; border-radius: 10px; padding: 2px 8px;
}
.step-facts { color: #909399; }

.tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; max-width: 78%; }
.link-tag { color: #606266; }

.disclaimer {
  max-width: 78%; margin-top: 6px; padding: 8px 12px;
  background: #fdf6ec; border: 1px solid #f5dab1; border-radius: 6px;
  color: #b88230; font-size: 12px; line-height: 1.6;
}
.meta { font-size: 11px; color: #c0c4cc; margin-top: 4px; }
</style>
