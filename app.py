import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# Page configuration is set in the main() function

def parse_expression(expr_str):
    """Parse a string mathematical expression into a SymPy expression.
    Handles common alternative syntax formats, including standard mathematical notation."""
    try:
        # Clean up the input string
        expr_str = expr_str.strip()
        
        # Check if it's a standard function format like "f(x) = 3x^2 + 2x - 5"
        if "f(x)" in expr_str or "f (x)" in expr_str:
            # Extract the part after the equals sign
            parts = expr_str.split("=", 1)
            if len(parts) > 1:
                expr_str = parts[1].strip()
        
        # Replace ^ with ** for exponentiation
        expr_str = expr_str.replace('^', '**')
        
        # Handle implicit multiplication with numbers (3x -> 3*x)
        import re
        # Match a number followed by a variable without an operator in between
        expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
        
        # Handle superscript numbers for exponents (x² -> x**2)
        superscript_map = {
            '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
            '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
        }
        for sup, num in superscript_map.items():
            expr_str = expr_str.replace(sup, f'**{num}')
        
        # Replace common trig functions if not already using sympy format
        replacements = {
            'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
            'asin': 'asin', 'arcsin': 'asin',
            'acos': 'acos', 'arccos': 'acos',
            'atan': 'atan', 'arctan': 'atan',
            'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh',
            'ln': 'log', 'log': 'log'
        }
        
        for old, new in replacements.items():
            # Only replace if it looks like a function call
            if f"{old}(" in expr_str and not f"sp.{old}(" in expr_str:
                expr_str = expr_str.replace(f"{old}(", f"sp.{new}(")
        
        # Handle sqrt function
        if "sqrt(" in expr_str and not "sp.sqrt(" in expr_str:
            expr_str = expr_str.replace("sqrt(", "sp.sqrt(")
            
        # Replace e with E for Euler's number if it's a standalone 'e'
        expr_str = expr_str.replace(' e ', ' E ')
        if expr_str == 'e':
            expr_str = 'E'
        
        # Handle common derivative notation d/dx
        if "d/dx" in expr_str:
            expr_str = expr_str.replace("d/dx", "")
            # Extract the function inside parentheses
            match = re.search(r'\((.*?)\)', expr_str)
            if match:
                expr_str = match.group(1)
                # Return the derivative directly
                x = sp.symbols('x')
                return sp.diff(sp.sympify(expr_str), x)
        
        return sp.sympify(expr_str)
    except Exception as e:
        st.error(f"Error parsing expression: {str(e)}")
        st.info("Try using standard formats like 'x^2', 'sin(x)', or even 'f(x) = 3x^2 + 2x - 5'. For more examples, see the sidebar.")
        return None

def plot_function(expr, x_range=(-10, 10), points=1000, is_derivative=False, is_integral=False, is_original=True):
    """Generate a plot for the given expression with improved styling."""
    x = sp.symbols('x')
    
    # Convert sympy expression to numpy function
    f = sp.lambdify(x, expr, "numpy")
    
    # Create x values
    x_vals = np.linspace(x_range[0], x_range[1], points)
    
    # Calculate y values, handling potential errors
    y_vals = []
    for x_val in x_vals:
        try:
            y_val = float(f(x_val))
            # Filter out very large values that would make the plot unusable
            if abs(y_val) > 1e10 or np.isnan(y_val) or np.isinf(y_val):
                y_val = None
            y_vals.append(y_val)
        except:
            y_vals.append(None)
    
    # Create a mask for valid values
    mask = [y is not None for y in y_vals]
    valid_x = x_vals[mask]
    valid_y = [y for y, m in zip(y_vals, mask) if m]
    
    # Determine plot color based on the type of function
    if is_derivative:
        line_color = '#FF5722'  # Orange for derivatives
        plot_title = f'Derivative: f\'(x) = {sp.latex(expr)}'
    elif is_integral:
        line_color = '#4CAF50'  # Green for integrals
        plot_title = f'Integral: ∫f(x)dx = {sp.latex(expr)} + C'
    else:
        line_color = '#2196F3'  # Blue for original functions
        plot_title = f'f(x) = {sp.latex(expr)}'
    
    # Create plot with improved styling
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(valid_x) > 0:
        ax.plot(valid_x, valid_y, color=line_color, linewidth=2.5)
    
    # Add a light background grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add x and y axes
    ax.axhline(y=0, color='#616161', linestyle='-', alpha=0.6, linewidth=1)
    ax.axvline(x=0, color='#616161', linestyle='-', alpha=0.6, linewidth=1)
    
    # Set better limits to avoid excessive white space
    if len(valid_y) > 0:
        y_min, y_max = min(valid_y), max(valid_y)
        y_range = y_max - y_min
        if y_range < 1e-10:  # Avoid division by zero or very small ranges
            y_range = 2
        ax.set_ylim([y_min - 0.1 * y_range, y_max + 0.1 * y_range])
    
    # Customize labels and title
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title(plot_title, fontsize=16, pad=10)
    
    # Add a light box around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDBDBD')
    
    # Make the figure background slightly off-white for better contrast
    fig.patch.set_facecolor('#F9F9F9')
    ax.set_facecolor('#F5F5F5')
    
    return fig

def main():
    """Main function for the Calculus Solver app."""
    # Set custom page metadata to remove Replit references
    st.set_page_config(
        page_title="KMSB Calculus Solver",
        page_icon="➗",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "# KMSB Calculus Solver\nA powerful tool for solving calculus problems."
        }
    )
    
    # Define styles for the entire application
    st.markdown("""
    <style>
    .result-box {
        background-color: #f0f7fb;
        border-left: 5px solid #2196F3;
        padding: 10px;
        margin: 20px 0;
        border-radius: 5px;
    }
    .input-function {
        background-color: #f8f9fa;
        border-left: 5px solid #6c757d;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .watermark {
        position: fixed;
        top: 20px;
        left: 20px;
        color: rgba(33, 150, 243, 0.5);  /* More visible blue color */
        font-size: 32px;  /* Larger font */
        font-weight: bold;
        font-family: 'Arial', sans-serif;
        z-index: 1000;
        transform: rotate(-5deg);
        letter-spacing: 3px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);  /* Add subtle shadow */
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add watermark with initials (two methods for reliability)
    st.markdown('<div class="watermark">KMSB</div>', unsafe_allow_html=True)
    
    # Backup watermark method using columns
    cols = st.columns([1, 10])
    with cols[0]:
        st.markdown('<span style="color: #2196F3; font-size: 20px; font-weight: bold;">KMSB</span>', unsafe_allow_html=True)
    
    st.title("Calculus Solver")
    
    st.markdown("""
    This application helps solve calculus problems:
    - Differentiation (find derivatives)
    - Integration (find antiderivatives)
    - Limits (evaluate limits)
    
    Enter a mathematical expression in terms of x below.
    """)
    
    # Create sidebar for operation selection
    with st.sidebar:
        st.header("Operation")
        operation = st.selectbox(
            "Choose Operation",
            ["Differentiate", "Integrate", "Limit"], 
            help="Select the calculus operation you want to perform on your function"
        )
        
        st.header("Examples")
        st.markdown("""
        **Functions:**
        - Standard format: `f(x) = 3x² + 2x - 5`
        - Polynomial: `x^2 + 3*x - 2` or `3x^2 + 2x - 5` 
        - Trigonometric: `sin(x) + cos(x)`
        - Logarithmic: `log(x)` or `ln(x)`
        - Exponential: `exp(x)` or `e^x`
        - Rational: `(x^2 - 4)/(x + 2)`
        - Square root: `sqrt(x)` or `x^(1/2)`
        
        **Constants & Symbols:**
        - π (pi): `pi`
        - e: `e` or `E` or `exp(1)`
        - Infinity: `oo` or `inf`
        
        **Syntax Tips:**
        - Standard notation like `f(x) = 3x² + 2x - 5` works directly
        - You can use superscript numbers (²) for powers
        - Implicit multiplication (3x) is automatically handled
        - Common math functions like `sin`, `cos`, `log` work directly
        """)
    
    # Main interface
    st.markdown("### Enter Your Function")
    expression = st.text_input("Enter function of x (e.g., f(x) = 3x² + 2x - 5)", 
                               help="You can enter in standard form like 'f(x) = 3x² + 2x - 5' or just the expression '3x² + 2x - 5'. Implicit multiplication like '3x' is supported.")
    
    if not expression:
        st.info("Enter a mathematical expression above to get started.")
        return
    
    # Define symbolic variable
    x = sp.symbols('x')
    
    # Parse the expression
    func = parse_expression(expression)
    
    if func is None:
        return
    
    # Display the entered function
    st.markdown("### Your Function:")
    st.markdown("<div class='input-function'>", unsafe_allow_html=True)
    st.latex(f"f(x) = {sp.latex(func)}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display current operation info
    operation_icons = {
        "Differentiate": "🔍 Differentiation",
        "Integrate": "🧮 Integration",
        "Limit": "🎯 Limit Evaluation"
    }
    
    st.markdown(f"## {operation_icons[operation]}")
    st.markdown(f"Analyzing: **`{expression}`**")
    
    # Show a plot of the function if possible
    st.markdown("### 📊 Function Graph")
    try:
        plot = plot_function(func, is_original=True, is_derivative=False, is_integral=False)
        st.pyplot(plot)
    except Exception as e:
        st.warning(f"Could not plot the function: {str(e)}")
    
    # Process based on selected operation
    if operation == "Differentiate":
        order = st.slider("Order of derivative", 1, 5, 1)
        
        try:
            result = func
            for _ in range(order):
                result = sp.diff(result, x)
                
            st.markdown(f"### 📝 Derivative (order {order}):")
            
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.latex(f"f^{{{order}}}(x) = {sp.latex(result)}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Try to plot the derivative
            try:
                st.markdown("### 📈 Derivative Graph")
                deriv_plot = plot_function(result, is_derivative=True, is_original=False, is_integral=False)
                st.pyplot(deriv_plot)
            except Exception as e:
                st.info(f"Could not plot the derivative: {str(e)}")
            
            # Simplify if possible
            simplified = sp.simplify(result)
            if simplified != result:
                st.markdown("### ✨ Simplified form:")
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.latex(f"f^{{{order}}}(x) = {sp.latex(simplified)}")
                st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error calculating derivative: {str(e)}")
    
    elif operation == "Integrate":
        integration_type = st.radio("Integration type:", ["Indefinite", "Definite"])
        
        try:
            if integration_type == "Indefinite":
                result = sp.integrate(func, x)
                st.markdown("### 📝 Indefinite Integral:")
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.latex(f"\\int {sp.latex(func)} \\, dx = {sp.latex(result)} + C")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Try to plot the integral
                try:
                    st.markdown("### 📈 Integral Graph")
                    integral_plot = plot_function(result, is_integral=True, is_original=False, is_derivative=False)
                    st.pyplot(integral_plot)
                except Exception as e:
                    st.info(f"Could not plot the integral: {str(e)}")
            
            else:  # Definite integral
                col1, col2 = st.columns(2)
                with col1:
                    lower_bound = st.text_input("Lower bound", "0")
                with col2:
                    upper_bound = st.text_input("Upper bound", "1")
                
                try:
                    lower = parse_expression(lower_bound)
                    upper = parse_expression(upper_bound)
                    
                    if lower is not None and upper is not None:
                        result = sp.integrate(func, (x, lower, upper))
                        st.markdown("### 📝 Definite Integral:")
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.latex(f"\\int_{{{sp.latex(lower)}}}^{{{sp.latex(upper)}}} {sp.latex(func)} \\, dx = {sp.latex(result)}")
                        st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error computing definite integral: {str(e)}")
        
        except Exception as e:
            st.error(f"Error calculating integral: {str(e)}")
    
    elif operation == "Limit":
        col1, col2 = st.columns(2)
        
        with col1:
            point = st.text_input("Point to evaluate limit (e.g., 0, oo, -oo)", "0")
        
        with col2:
            direction = st.selectbox("Direction", ["both", "+", "-"])
        
        try:
            point_val = parse_expression(point)
            
            if point_val is not None:
                if direction == "both":
                    result = sp.limit(func, x, point_val)
                else:
                    result = sp.limit(func, x, point_val, dir=direction)
                
                st.markdown("### 📝 Limit Result:")
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.latex(f"\\lim_{{x \\to {sp.latex(point_val)}{'+' if direction == '+' else '-' if direction == '-' else ''}}} {sp.latex(func)} = {sp.latex(result)}")
                st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error calculating limit: {str(e)}")

if __name__ == "__main__":
    main()