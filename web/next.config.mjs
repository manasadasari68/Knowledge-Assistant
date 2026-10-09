/** @type {import('next').NextConfig} */
const apiBaseUrl = (process.env.API_BASE_URL || "http://localhost:8000").replace(
  /\/+$/,
  "",
);

const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${apiBaseUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
