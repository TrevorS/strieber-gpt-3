<script lang="ts">
	import { goto } from '$app/navigation';
	import { ChatInput, MessageList } from '$lib/components/chat';
	import { sendMessageStreaming } from '$lib/api';
	import { conversationStore } from '$lib/stores';
	import { logger } from '$lib/utils/logger';

	let isStreaming = $state(false);

	// Explicitly track activeId to ensure reactivity when it changes to null
	let activeId = $derived(conversationStore.activeId);
	let activeConversation = $derived(activeId ? conversationStore.get(activeId) : undefined);
	let messages = $derived(activeConversation?.messages ?? []);

	logger.lifecycle.mount('HomePage', { activeId });

	async function handleSubmit(text: string) {
		logger.ui.event('HomePage', 'handleSubmit called', {
			textLength: text.length,
			hasActiveConversation: !!activeConversation
		});

		// Create conversation if needed
		let conv = activeConversation;
		if (!conv) {
			logger.info('ui', 'Creating new conversation for message');
			conv = conversationStore.create();
		}

		// Add user message
		conversationStore.addMessage(conv.id, 'user', text);

		// Create placeholder for assistant message
		const assistantMessage = conversationStore.addMessage(conv.id, 'assistant', '');
		conversationStore.setMessageStreaming(conv.id, assistantMessage.id, true);

		isStreaming = true;

		// Navigate to the conversation URL
		logger.nav.navigate('/', `/c/${conv.id}`, { conversationId: conv.id });
		goto(`/c/${conv.id}`);

		// Stream the response
		logger.api.request('POST', '/responses', { previousResponseId: conv.lastResponseId });

		await sendMessageStreaming(
			text,
			{
				previousResponseId: conv.lastResponseId
			},
			{
				onDelta: (content) => {
					conversationStore.updateMessageContent(conv!.id, assistantMessage.id, content);
				},
				onOutputItem: (item) => {
					conversationStore.setOutputItem(conv!.id, assistantMessage.id, item);
				},
				onComplete: (responseId) => {
					logger.api.streamComplete(conv!.id, conversationStore.get(conv!.id)?.messages.find(m => m.id === assistantMessage.id)?.content.length ?? 0);
					conversationStore.updateLastResponseId(conv!.id, responseId);
					conversationStore.setMessageStreaming(conv!.id, assistantMessage.id, false);
					isStreaming = false;
				},
				onError: (error) => {
					logger.error('api', 'Stream error', { conversationId: conv!.id, error: error.message });
					conversationStore.updateMessageContent(
						conv!.id,
						assistantMessage.id,
						`Error: ${error.message}`
					);
					conversationStore.setMessageStreaming(conv!.id, assistantMessage.id, false);
					isStreaming = false;
				}
			}
		);
	}
</script>

<MessageList {messages} />
<ChatInput onsubmit={handleSubmit} disabled={isStreaming} />
