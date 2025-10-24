import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Providers } from './providers';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Solar Rooftop Analysis - AI-Powered Solar Assessment',
  description: 'AI-powered solar rooftop analysis with cutting-edge technology including vision transformers, physics-informed AI, and AR visualization.',
  keywords: 'solar, rooftop, analysis, AI, renewable energy, sustainability',
  authors: [{ name: 'Solar Analysis Team' }],
  viewport: 'width=device-width, initial-scale=1',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  );
}
