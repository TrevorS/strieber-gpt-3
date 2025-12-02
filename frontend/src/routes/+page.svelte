<script lang="ts">
	import { ChatInput, MessageList } from '$lib/components/chat';
	import { sendMessageStreaming } from '$lib/api';
	import { conversationStore } from '$lib/stores';

	let isStreaming = $state(false);

	// Get messages from the active conversation
	let messages = $derived(conversationStore.active?.messages ?? []);

	async function handleSubmit(text: string) {
		// Create conversation if needed
		let conv = conversationStore.active;
		if (!conv) {
			conv = conversationStore.create();
		}

		// Add user message
		conversationStore.addMessage(conv.id, 'user', text);

		// Create placeholder for assistant message
		const assistantMessage = conversationStore.addMessage(conv.id, 'assistant', '');
		conversationStore.setMessageStreaming(conv.id, assistantMessage.id, true);

		isStreaming = true;

		// Stream the response
		await sendMessageStreaming(
			text,
			{
				previousResponseId: conv.lastResponseId
			},
			{
				onDelta: (content) => {
					conversationStore.updateMessageContent(conv!.id, assistantMessage.id, content);
				},
				onComplete: (responseId) => {
					conversationStore.updateLastResponseId(conv!.id, responseId);
					conversationStore.setMessageStreaming(conv!.id, assistantMessage.id, false);
					isStreaming = false;
				},
				onError: (error) => {
					console.error('Stream error:', error);
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
