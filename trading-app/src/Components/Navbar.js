import React, { useState,useContext,useEffect} from 'react';
import { Home, Brain, Zap, Menu, X, User, LogOut, LogIn , Sun , Moon ,History } from 'lucide-react';
import '../Styles/Navbar.css';
import NoteContext from '../Context/NoteContext';
import { useNavigate } from 'react-router-dom';
import { useLocation } from 'react-router-dom';
const pathToItemId = (pathname) => {
  if (pathname === '/') return 'home';
  if (pathname.startsWith('/train')) return 'trainer';
  if (pathname.startsWith('/options')) return 'simulation';
  if (pathname.startsWith('/pastanalysis')) return 'pastanalysis';
  return 'home'; // fallback
};
const Navbar = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const {username, setUsername,isAuthenticated, setIsAuthenticated,darkMode,email,setDarkMode,activeItem, setActiveItem}= useContext(NoteContext);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const navigate = useNavigate() ; 
  const toggleTheme = () => {
    setDarkMode(!darkMode);
  };
  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };
  
  useEffect(() => {
  const theme = darkMode ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', theme);
}, [darkMode]);
const location = useLocation();
/*{ id: 'trainer', label: 'Model Trainer', icon: Brain , link:'/train' }*/
useEffect(() => {
  const matchedId = pathToItemId(location.pathname);
  setActiveItem(matchedId);
}, [location]);
  const navItems = [
    { id: 'home', label: 'Home Page', icon: Home , link:'/' },
    { id: 'simulation', label: 'AI Simulation', icon: Zap , link:'/options' },
    { id: 'pastanalysis', label: 'Past Analysis', icon: History , link:'/pastanalysis' },
  ];

  const handleNavClick = (itemId) => {
    setActiveItem(itemId);
    setIsMobileMenuOpen(false);
  };

  const handleLogin = () => {
    // Simulate login - in real app, this would handle actual authentication
    navigate('/signin');
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setUsername('');
    setShowUserMenu(false);
  };

  const toggleUserMenu = () => {
    setShowUserMenu(!showUserMenu);
  };

  const closeUserMenu = () => {
    setShowUserMenu(false);
  };

  const getInitials = (name) => {
    return name
      .split(' ')
      .map(word => word.charAt(0))
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  return (
    <div className={`navbar-wrapper ${darkMode ? 'dark' : ''}`}>
      <nav className="navbar">
        <div className="navbar-container">
          {/* Logo */}
          <div className="logo">
            <div className="logo-icon" onClick={() => navigate('/')}>
              <Brain size={20} />
            </div>
            <span>DerivIQ</span>
          </div>

          {/* Desktop Navigation */}
          <div className="nav-items">
            {navItems.map((item) => {
              const IconComponent = item.icon;
              return (
                <button
                  key={item.id}
                  onClick={() => {handleNavClick(item.id)
                    navigate(item.link)}
                  }
                  className={`nav-item ${activeItem === item.id ? 'active' : ''}`}
                >
                  <IconComponent size={18} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </div>

          {/* Theme Toggle & User Auth & Mobile Menu */}
          <div className="navbar-controls">
            {/* Theme Toggle */}
            <div className="theme-toggle">
              <span className="theme-label"><Sun color = {darkMode? '#ffffff' : '#000000'} size={16}/></span>
              <button 
                onClick={toggleTheme}
                className="toggle-switch"
                aria-label="Toggle dark mode"
              >
                <div className="toggle-knob"></div>
              </button>
              <span className="theme-label"><Moon color = {darkMode? '#ffffff' : '#000000'} size={16}/></span>
            </div>
            {/* User Authentication */}
            {isAuthenticated ? (
              <div className="user-section">
                 <button 
                    onClick={toggleUserMenu}
                    className="user-avatar"
                    aria-label="User menu"
                  >
                    {getInitials(username)}
                  </button>
                <span className="username">{username}</span>
                <div className="user-menu-container">
                 
                  
                  {showUserMenu && (
                    <>
                      <div className="overlay" onClick={closeUserMenu}></div>
                      <div className="user-dropdown">
                        <div className="user-info">
                          <div className="user-avatar-large">
                            {getInitials(username)}
                          </div>
                          <div>
                            <div className="user-name">{username}</div>
                            <div className="user-email">{email ? email : 'example@gmail.com'}</div>
                          </div>
                        </div>
                        <hr className="dropdown-divider" />
                        <button 
                          onClick={handleLogout}
                          className="dropdown-item logout-btn"
                        >
                          <LogOut size={16} />
                          <span>Logout</span>
                        </button>
                      </div>
                    </>
                  )}
                </div>
              </div>
            ) : (
              <button 
                onClick={handleLogin}
                className="login-btn"
                aria-label="Login"
              >
                <LogIn size={18} />
                <span>Login</span>
              </button>
            )}
            {/* Mobile Menu Button */}
            <button
              onClick={toggleMobileMenu}
              className="mobile-menu-button"
              aria-label="Toggle mobile menu"
            >
              {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        <div className={`mobile-menu ${isMobileMenuOpen ? 'open' : ''}`}>
          {navItems.map((item) => {
            const IconComponent = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => handleNavClick(item.id)}
                className={`mobile-nav-item ${activeItem === item.id ? 'active' : ''}`}
              >
                <IconComponent size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </nav>
    </div>
  );
};

export default Navbar;