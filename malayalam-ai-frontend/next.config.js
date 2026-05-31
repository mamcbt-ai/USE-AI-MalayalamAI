/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://use-ai-malayalamai-production-ee70.up.railway.app/:path*',
      },
    ]
  },
}

module.exports = nextConfig
