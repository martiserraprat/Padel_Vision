'use client'

import { useState, useEffect } from 'react'
import Modal from '@/components/ui/Modal'

const C = {
  green:     '#b8f53d',
  greenDark: '#8ec22a',
  black:     '#0a0a0a',
  cream:     '#f5f0e8',
  gray1:     '#111111',
  gray2:     '#1a1a1a',
  gray3:     '#2a2a2a',
  muted:     '#666666',
}

function InputField({ label, type = 'text', placeholder }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <label style={{
        fontSize: '10px',
        textTransform: 'uppercase',
        letterSpacing: '0.15em',
        color: C.muted,
        fontFamily: "'DM Sans', sans-serif",
      }}>
        {label}
      </label>
      <input
        type={type}
        placeholder={placeholder}
        style={{
          width: '100%',
          background: C.black,
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: '8px',
          padding: '12px 16px',
          color: C.cream,
          fontSize: '14px',
          fontFamily: "'DM Sans', sans-serif",
          outline: 'none',
          transition: 'border-color 0.2s',
        }}
        onFocus={e => e.target.style.borderColor = 'rgba(184,245,61,0.5)'}
        onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.1)'}
      />
    </div>
  )
}

export default function AuthModal({ isOpen, onClose, defaultTab = 'login' }) {
  const [tab, setTab] = useState(defaultTab)

  // Quan s'obre el modal, actualitza el tab al que toca
  useEffect(() => {
    if (isOpen) setTab(defaultTab)
  }, [isOpen, defaultTab])

  const tabs = [
    { id: 'login',    label: 'Iniciar sessió' },
    { id: 'register', label: 'Registrar-se'   },
  ]

  return (
    <Modal isOpen={isOpen} onClose={onClose}>

      <h2 style={{
        fontFamily: "'Bebas Neue', sans-serif",
        fontSize: '38px',
        letterSpacing: '0.05em',
        color: C.cream,
        marginBottom: '4px',
      }}>
        {tab === 'login' ? 'BENVINGUT' : "REGISTRA'T"}
      </h2>
      <p style={{ color: C.muted, fontSize: '13px', marginBottom: '24px' }}>
        {tab === 'login'
          ? 'Accedeix al teu compte de PadelVision'
          : 'Crea el teu compte gratis'}
      </p>

      <div style={{
        display: 'flex',
        gap: '4px',
        background: C.black,
        borderRadius: '10px',
        padding: '4px',
        marginBottom: '24px',
      }}>
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              flex: 1,
              borderRadius: '7px',
              padding: '8px',
              fontSize: '12px',
              fontFamily: "'DM Sans', sans-serif",
              fontWeight: 500,
              border: 'none',
              cursor: 'pointer',
              transition: 'all 0.2s',
              background: tab === t.id ? C.gray3 : 'transparent',
              color: tab === t.id ? C.cream : C.muted,
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      <form
        onSubmit={(e) => e.preventDefault()}
        style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}
      >
        {tab === 'register' && (
          <InputField label="Nom complet" placeholder="Martí Serra" />
        )}
        <InputField label="Correu electrònic" type="email" placeholder="marti@example.com" />
        <InputField label="Contrasenya" type="password" placeholder="••••••••" />

        {tab === 'login' && (
          <p
            style={{ textAlign: 'right', fontSize: '12px', color: C.muted, cursor: 'pointer', marginTop: '-8px' }}
            onMouseEnter={e => e.target.style.color = C.cream}
            onMouseLeave={e => e.target.style.color = C.muted}
          >
            Has oblidat la contrasenya?
          </p>
        )}

        <button
          type="submit"
          style={{
            width: '100%',
            background: C.green,
            color: C.black,
            border: 'none',
            borderRadius: '8px',
            padding: '14px',
            fontSize: '14px',
            fontWeight: 600,
            fontFamily: "'DM Sans', sans-serif",
            cursor: 'pointer',
            marginTop: '4px',
            transition: 'background 0.2s',
          }}
          onMouseEnter={e => e.target.style.background = C.greenDark}
          onMouseLeave={e => e.target.style.background = C.green}
        >
          {tab === 'login' ? 'Iniciar sessió' : 'Crear compte'}
        </button>
      </form>

      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: '20px 0' }}>
        <div style={{ flex: 1, height: '1px', background: 'rgba(255,255,255,0.1)' }} />
        <span style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.15em', color: C.muted }}>o</span>
        <div style={{ flex: 1, height: '1px', background: 'rgba(255,255,255,0.1)' }} />
      </div>

      <p style={{ textAlign: 'center', fontSize: '12px', color: C.muted }}>
        {tab === 'login' ? "Encara no tens compte? " : 'Ja tens compte? '}
        <button
          onClick={() => setTab(tab === 'login' ? 'register' : 'login')}
          style={{
            background: 'none',
            border: 'none',
            color: C.green,
            cursor: 'pointer',
            fontSize: '12px',
            fontFamily: "'DM Sans', sans-serif",
            textDecoration: 'underline',
          }}
        >
          {tab === 'login' ? "Registra't" : 'Inicia sessió'}
        </button>
      </p>
    </Modal>
  )
}