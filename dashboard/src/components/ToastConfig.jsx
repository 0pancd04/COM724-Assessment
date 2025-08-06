import React from 'react';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

export const ToastConfig = () => (
  <ToastContainer
    position="bottom-right"
    autoClose={2000}
    hideProgressBar={false}
    newestOnTop
    closeOnClick
    rtl={false}
    pauseOnFocusLoss
    draggable
    pauseOnHover
    theme="light"
    icon={({ type }) => {
      switch (type) {
        case 'success':
          return <span>✓</span>;
        case 'error':
          return <span>✕</span>;
        case 'info':
          return <span>ℹ</span>;
        case 'warning':
          return <span>⚠</span>;
        default:
          return null;
      }
    }}
  />
);
