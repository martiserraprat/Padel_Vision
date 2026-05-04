'use client'

import { useState } from 'react'
import Button from '@/components/ui/Button'
import AuthModal from '@/components/ui/AuthModal'

export default function CtaSection() {
  const [modalOpen, setModalOpen] = useState(false)

  return (
    <>
      <section className="border-t border-white/7 px-8 py-32 text-center">
        <h2
          className="font-display leading-[0.88] mb-6 text-cream"
          style={{ fontSize: 'clamp(48px,8vw,112px)' }}
        >
          ANALITZA<br />
          EL TEU <span className="text-green">PROPER</span><br />
          PARTIT
        </h2>
        <p className="text-muted text-base mb-10 font-light">
          Registra't gratis i puja el teu primer vídeo en menys d'un minut.
        </p>
        <Button variant="primary" size="lg" onClick={() => setModalOpen(true)}>
          Crear compte gratis
        </Button>
      </section>

      <AuthModal isOpen={modalOpen} onClose={() => setModalOpen(false)} defaultTab="register" />
    </>
  )
}
