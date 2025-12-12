/**
 * VSCode-style resizable sidebar
 * Handles drag-to-resize interaction for the file explorer sidebar
 */

(function() {
    'use strict';
    
    // Configuration
    const MIN_WIDTH = 180;
    const MAX_WIDTH = 600;
    const DEFAULT_WIDTH = 280;
    const STORAGE_KEY = 'explorer-width-store';
    
    // State
    let isResizing = false;
    let startX = 0;
    let startWidth = 0;
    let isInitialized = false;
    let currentResizer = null;
    let eventHandlers = {
        mousedown: null,
        mousemove: null,
        mouseup: null,
        selectstart: null
    };
    let observer = null;
    let lastUrl = window.location.pathname;
    
    /**
     * Clean up event listeners
     */
    function cleanup() {
        if (currentResizer) {
            if (eventHandlers.mousedown) {
                currentResizer.removeEventListener('mousedown', eventHandlers.mousedown);
            }
            if (eventHandlers.selectstart) {
                currentResizer.removeEventListener('selectstart', eventHandlers.selectstart);
            }
        }
        
        if (eventHandlers.mousemove) {
            document.removeEventListener('mousemove', eventHandlers.mousemove);
        }
        if (eventHandlers.mouseup) {
            document.removeEventListener('mouseup', eventHandlers.mouseup);
        }
        
        // Reset handlers
        eventHandlers = {
            mousedown: null,
            mousemove: null,
            mouseup: null,
            selectstart: null
        };
        
        currentResizer = null;
        isInitialized = false;
    }
    
    /**
     * Apply saved width immediately (before full initialization)
     * This prevents the flash of default width
     */
    function applyWidthImmediately() {
        const explorerCol = document.getElementById('file-explorer-col');
        if (explorerCol) {
            try {
                const storedData = localStorage.getItem(STORAGE_KEY);
                if (storedData) {
                    const width = JSON.parse(storedData);
                    if (typeof width === 'number' && width >= MIN_WIDTH && width <= MAX_WIDTH) {
                        // Apply width immediately using inline styles
                        explorerCol.style.flex = `0 0 ${width}px`;
                        explorerCol.style.width = `${width}px`;
                        return true;
                    }
                }
            } catch (e) {
                // Ignore errors during immediate application
            }
        }
        return false;
    }
    
    /**
     * Initialize the resizer when the DOM is ready
     */
    function initializeResizer() {
        const resizer = document.getElementById('resizer');
        const explorerCol = document.getElementById('file-explorer-col');
        
        if (!resizer || !explorerCol) {
            // Elements not found yet
            return false;
        }
        
        // If already initialized for this resizer, skip
        if (isInitialized && currentResizer === resizer) {
            return true;
        }
        
        // Clean up previous initialization if any
        cleanup();
        
        // Load saved width from localStorage (may have been applied already, but ensure it's set)
        loadSavedWidth();
        
        // Create bound event handlers
        eventHandlers.mousedown = handleMouseDown.bind(null);
        eventHandlers.mousemove = handleMouseMove.bind(null);
        eventHandlers.mouseup = handleMouseUp.bind(null);
        eventHandlers.selectstart = (e) => e.preventDefault();
        
        // Add event listeners
        resizer.addEventListener('mousedown', eventHandlers.mousedown);
        document.addEventListener('mousemove', eventHandlers.mousemove);
        document.addEventListener('mouseup', eventHandlers.mouseup);
        resizer.addEventListener('selectstart', eventHandlers.selectstart);
        
        // Track current resizer and initialization state
        currentResizer = resizer;
        isInitialized = true;
        
        return true;
    }
    
    /**
     * Load saved width from localStorage
     */
    function loadSavedWidth() {
        try {
            const storedData = localStorage.getItem(STORAGE_KEY);
            if (storedData) {
                const width = JSON.parse(storedData);
                if (typeof width === 'number' && width >= MIN_WIDTH && width <= MAX_WIDTH) {
                    setExplorerWidth(width);
                }
            }
        } catch (e) {
            console.warn('Failed to load saved explorer width:', e);
        }
    }
    
    /**
     * Save width to localStorage
     */
    function saveWidth(width) {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(width));
            
            // Also update Dash store if it exists
            const storeElement = document.getElementById('explorer-width-store');
            if (storeElement && window.dash_clientside) {
                // Trigger a custom event that Dash can listen to
                const event = new CustomEvent('explorer-width-changed', {
                    detail: { width: width }
                });
                document.dispatchEvent(event);
            }
        } catch (e) {
            console.warn('Failed to save explorer width:', e);
        }
    }
    
    /**
     * Set the explorer column width
     */
    function setExplorerWidth(width) {
        const explorerCol = document.getElementById('file-explorer-col');
        if (explorerCol) {
            explorerCol.style.flex = `0 0 ${width}px`;
            explorerCol.style.width = `${width}px`;
        }
    }
    
    /**
     * Handle mouse down on resizer
     */
    function handleMouseDown(e) {
        e.preventDefault();
        isResizing = true;
        startX = e.clientX;
        
        const explorerCol = document.getElementById('file-explorer-col');
        const resizer = document.getElementById('resizer');
        
        if (explorerCol) {
            startWidth = explorerCol.offsetWidth;
        }
        
        if (resizer) {
            resizer.classList.add('resizing');
        }
        
        // Add class to body to prevent text selection
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    }
    
    /**
     * Handle mouse move during resize
     */
    function handleMouseMove(e) {
        if (!isResizing) return;
        
        e.preventDefault();
        
        const deltaX = e.clientX - startX;
        let newWidth = startWidth + deltaX;
        
        // Clamp width between min and max
        newWidth = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, newWidth));
        
        setExplorerWidth(newWidth);
    }
    
    /**
     * Handle mouse up (end resize)
     */
    function handleMouseUp(e) {
        if (!isResizing) return;
        
        isResizing = false;
        
        const resizer = document.getElementById('resizer');
        if (resizer) {
            resizer.classList.remove('resizing');
        }
        
        // Restore body cursor and selection
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        
        // Save the new width
        const explorerCol = document.getElementById('file-explorer-col');
        if (explorerCol) {
            const finalWidth = explorerCol.offsetWidth;
            saveWidth(finalWidth);
        }
    }
    
    /**
     * Handle element detection and immediate width application
     */
    function handleElementDetected() {
        const explorerCol = document.getElementById('file-explorer-col');
        const resizer = document.getElementById('resizer');
        
        // Double-check elements exist
        if (!explorerCol || !resizer) {
            return;
        }
        
        // Check if already initialized for this resizer
        if (isInitialized && currentResizer === resizer) {
            return;
        }
        
        const currentUrl = window.location.pathname;
        lastUrl = currentUrl;
        
        // First, immediately apply width to prevent flash
        // This happens synchronously, before any rendering delay
        applyWidthImmediately();
        
        // Then initialize resizer functionality
        // Use requestAnimationFrame to ensure DOM is ready
        requestAnimationFrame(() => {
            initializeResizer();
        });
    }
    
    /**
     * Setup MutationObserver to detect file-explorer-col element
     * This fires immediately when the element is added to DOM
     */
    function setupObserver() {
        // Disconnect existing observer if any
        if (observer) {
            observer.disconnect();
        }
        
        // Create new observer that watches for file-explorer-col specifically
        observer = new MutationObserver(() => {
            // Use a simple, fast check - just see if elements exist
            const explorerCol = document.getElementById('file-explorer-col');
            const resizer = document.getElementById('resizer');
            
            if (explorerCol && resizer) {
                // Elements found, check if we need to initialize
                if (!isInitialized || currentResizer !== resizer) {
                    // Handle immediately - apply width first, then initialize
                    handleElementDetected();
                }
            } else if (isInitialized && (!explorerCol || !resizer)) {
                // Elements were removed (page navigation), cleanup
                cleanup();
            }
        });
        
        // Observe the entire document for changes
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
    
    // Initialize when DOM is ready
    function tryInitialize() {
        // First try to apply width immediately if element exists
        if (applyWidthImmediately()) {
            // Element exists, initialize fully
            initializeResizer();
        }
        
        // Setup observer for future changes
        setupObserver();
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', tryInitialize);
    } else {
        // DOM is already ready
        tryInitialize();
    }
    
    /**
     * Handle URL change - check immediately if elements exist
     */
    function handleUrlChange() {
        const currentUrl = window.location.pathname;
        if (currentUrl !== lastUrl) {
            lastUrl = currentUrl;
            
            // Immediately check if elements already exist (fast path)
            // Try to apply width synchronously first to prevent flash
            const explorerCol = document.getElementById('file-explorer-col');
            if (explorerCol) {
                // Element exists, apply width immediately (synchronously)
                applyWidthImmediately();
                
                // Then check for resizer and initialize
                const resizer = document.getElementById('resizer');
                if (resizer) {
                    // Both elements exist, initialize fully
                    requestAnimationFrame(() => {
                        initializeResizer();
                    });
                }
            }
            // If elements don't exist yet, MutationObserver will catch them when they appear
        }
    }
    
    // Listen for URL changes (Dash navigation and browser back/forward)
    // Dash uses pushState, so we intercept it
    const originalPushState = history.pushState;
    history.pushState = function() {
        originalPushState.apply(history, arguments);
        // Check immediately (synchronously) to apply width before render
        handleUrlChange();
        // Also check after next frame in case elements appear later
        requestAnimationFrame(() => {
            handleUrlChange();
        });
    };
    
    window.addEventListener('popstate', () => {
        // Check immediately
        handleUrlChange();
        // Also check after next frame
        requestAnimationFrame(() => {
            handleUrlChange();
        });
    });
    
})();

