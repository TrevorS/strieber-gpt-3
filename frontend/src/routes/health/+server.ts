/**
 * Health Check Endpoint
 *
 * Returns a simple JSON response for container health checks.
 * Used by Docker HEALTHCHECK and load balancers.
 */

import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => {
	return json({
		status: 'healthy',
		timestamp: new Date().toISOString()
	});
};
