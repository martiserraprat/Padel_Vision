'use client'

import { useState, useEffect } from 'react'
import Button from '@/components/ui/Button'
import AuthModal from '@/components/ui/AuthModal'

export default function Navbar() {
  const [scrolled, setScrolled]   = useState(false)
  const [modalOpen, setModalOpen] = useState(false)
  const [modalTab, setModalTab]   = useState('login')

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', handler)
    return () => window.removeEventListener('scroll', handler)
  }, [])

  const openLogin    = () => { setModalTab('login');    setModalOpen(true) }
  const openRegister = () => { setModalTab('register'); setModalOpen(true) }

  return (
    <>
      <nav
        className={`fixed inset-x-0 top-0 z-40 flex items-center justify-between px-8 transition-all duration-300
          ${scrolled
            ? 'py-4 border-b border-white/7 bg-black/90 backdrop-blur-xl'
            : 'py-5 bg-transparent'
          }`}
      >
        <a href="/" className="font-display text-2xl tracking-widest text-cream">
          Padel<span className="text-green">Vision</span>
        </a>

        <ul className="hidden md:flex items-center gap-8 list-none">
          {[
            { href: '#how',      label: 'Com funciona' },
            { href: '#features', label: 'Funcionalitats' },
          ].map((link) => (
            <li key={link.href}>
              <a
                href={link.href}
                className="text-xs uppercase tracking-widest text-muted hover:text-cream transition-colors duration-200"
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>

        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={openLogin}>Iniciar sessió</Button>
          <Button variant="primary" size="sm" onClick={openRegister}>Registrar-se</Button>
        </div>
      </nav>

      <AuthModal isOpen={modalOpen} onClose={() => setModalOpen(false)} defaultTab={modalTab} />
    </>
  )
}
