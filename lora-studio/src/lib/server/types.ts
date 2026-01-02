// Types matching lora-trainer REST API responses

export interface Dataset {
	name: string;
	trigger_token: string;
	lora_type: 'character' | 'style' | 'concept';
	description: string | null;
	image_count: number;
	has_captions: boolean;
	created_at: string;
}

export interface DatasetImage {
	filename: string;
	caption: string | null;
}

export interface DatasetDetail extends Dataset {
	images: DatasetImage[];
}

export interface AddImagesResult {
	added: number;
	failed: number;
	total_images: number;
	failures: Array<{ url: string; error: string }>;
}

export interface CaptionResult {
	captioned: number;
	failed?: number;
	message?: string;
	results?: Array<{ filename: string; caption?: string; error?: string }>;
}
