import './globals.css'

export const metadata = {
  title: 'PadelVision — Anàlisi tàctic amb visió per computador',
  description: 'Puja un vídeo del teu partit de pàdel i obté un mapa de calor del moviment dels jugadors.',
}

export default function RootLayout({ children }) {
  return (
    <html lang="ca">
      <body>{children}</body>
    </html>
  )
}
