'use client'

import { useState, useEffect } from 'react'
import Button from '@/components/ui/Button'
import AuthModal from '@/components/ui/AuthModal'

function StatCounter({ target, label, delay = 0 }) {
  const [value, setValue] = useState(0)

  useEffect(() => {
    const timeout = setTimeout(() => {
      const steps = 45, duration = 1800
      let step = 0
      const timer = setInterval(() => {
        step++
        const ease = 1 - Math.pow(1 - step / steps, 3)
        setValue(Math.round(target * ease))
        if (step >= steps) clearInterval(timer)
      }, duration / steps)
      return () => clearInterval(timer)
    }, delay)
    return () => clearTimeout(timeout)
  }, [target, delay])

  return (
    <div className="flex flex-col gap-1">
      <span className="font-display text-5xl text-green leading-none">
        {value.toLocaleString('ca')}
      </span>
      <span className="text-[10px] uppercase tracking-widest text-muted">
        {label}
      </span>
    </div>
  )
}

export default function Hero() {
  const [modalOpen, setModalOpen] = useState(false)
  const [modalTab, setModalTab]   = useState('register')

  const openRegister = () => { setModalTab('register'); setModalOpen(true) }
  const openLogin    = () => { setModalTab('login');    setModalOpen(true) }

  return (
    <>
      <section className="relative flex min-h-screen flex-col justify-center overflow-hidden px-8 pb-20 pt-32">

        {/* Grid bg */}
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            backgroundImage:
              'linear-gradient(rgba(184,245,61,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(184,245,61,0.04) 1px, transparent 1px)',
            backgroundSize: '64px 64px',
            maskImage: 'radial-gradient(ellipse 75% 65% at 40% 50%, black 20%, transparent 100%)',
          }}
        />

        {/* Court wireframe */}
        <div
          className="pointer-events-none absolute right-0 top-1/2 opacity-30 hidden lg:block"
          style={{ transform: 'translateY(-50%) perspective(900px) rotateY(-18deg) rotateX(8deg)' }}
        >
          <svg width="500" height="340" viewBox="0 0 500 340" fill="none">
            <rect x="2" y="2" width="496" height="336" stroke="rgba(184,245,61,0.4)" strokeWidth="2" rx="2"/>
            <line x1="250" y1="2" x2="250" y2="338" stroke="rgba(184,245,61,0.3)" strokeWidth="1.5"/>
            <line x1="2" y1="170" x2="498" y2="170" stroke="rgba(184,245,61,0.2)" strokeWidth="1"/>
            <rect x="2" y="85" width="496" height="170" stroke="rgba(184,245,61,0.15)" strokeWidth="1"/>
          </svg>
        </div>

        <div className="relative max-w-2xl">

          {/* Badge */}
          <div
            className="mb-8 inline-flex items-center gap-2 rounded-full border border-green/30 bg-green/10 px-4 py-2 animate-fade-up"
            style={{ animationDelay: '0ms' }}
          >
            <span className="animate-pulse-dot h-1.5 w-1.5 rounded-full bg-green" />
            <span className="text-[10px] uppercase tracking-[0.2em] text-green">
              Visió per Computador · UAB 2025
            </span>
          </div>

          {/* Heading */}
          <h1
            className="font-display leading-[0.88] mb-7 animate-fade-up text-cream"
            style={{
              fontSize: 'clamp(72px, 11vw, 148px)',
              animationDelay: '80ms',
              opacity: 0,
            }}
          >
            ANALISI<br />
            <span className="text-green">TACTIC</span><br />
            EN PADEL
          </h1>

          {/* Subtitle */}
          <p
            className="text-base text-muted leading-relaxed max-w-md mb-10 animate-fade-up"
            style={{ animationDelay: '180ms', opacity: 0 }}
          >
            Puja un vídeo del teu partit i obté un mapa de calor del moviment dels jugadors.
            Anàlisi automàtica amb visió per computador clàssica.
          </p>

          {/* Actions */}
          <div
            className="flex items-center gap-4 mb-16 animate-fade-up"
            style={{ animationDelay: '280ms', opacity: 0 }}
          >
            <Button variant="primary" size="lg" onClick={openRegister}>Comença gratis</Button>
            <Button variant="ghost"   size="lg" onClick={openLogin}>Ja tinc compte</Button>
          </div>

          {/* Stats */}
          <div
            className="flex gap-10 border-t border-white/7 pt-8 animate-fade-up"
            style={{ animationDelay: '400ms', opacity: 0 }}
          >
            <StatCounter target={1240} label="Vídeos analitzats" delay={500} />
            <StatCounter target={4860} label="Jugadors detectats" delay={600} />
            <StatCounter target={320}  label="Partits processats" delay={700} />
          </div>
        </div>
      </section>

      <AuthModal isOpen={modalOpen} onClose={() => setModalOpen(false)} defaultTab={modalTab} />
    </>
  )
}
