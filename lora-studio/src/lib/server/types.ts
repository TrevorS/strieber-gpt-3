// Types matching mcp-lora-trainer responses

export interface Dataset {
	name: string;
	path: string;
	image_count: number;
	has_captions: boolean;
	created_at?: string;
}

export interface DatasetInfo {
	name: string;
	path: string;
	images: DatasetImage[];
}

export interface DatasetImage {
	filename: string;
	path: string;
	caption: string | null;
	caption_file: string | null;
}

export interface AddImagesResult {
	added: string[];
	failed: Array<{ source: string; error: string }>;
}

export interface CaptionResult {
	captioned: number;
	skipped: number;
	failed: number;
	details: Array<{ image: string; status: string; caption?: string }>;
}
