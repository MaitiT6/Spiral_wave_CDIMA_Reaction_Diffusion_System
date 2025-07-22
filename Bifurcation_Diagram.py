import numpy as np
import matplotlib.pyplot as plt

def calculate_b_hopf(a, sigma):
    return (1 / sigma) * ((3 * a / 5) - (25 / a))

def calculate_b_tur(a, d):
    u = a / 5.0
    x = u * u
    y = d * u * (10 + 26 * u * u)
    z = d * d * ((3 * u * u) - 5) ** 2
    discriminant = y ** 2 - 4 * x * z
    
    if discriminant >= 0:
        return (y - np.sqrt(discriminant)) / (2 * x)
    return None

def main():
    sigma = 1.0
    d = 1.0  
    h = 0.002
    n = 20000

    a_values = []
    b_hopf_values = []
    b_tur_values = []

    a = 0.001
    for _ in range(n):
        b_hopf = calculate_b_hopf(a, sigma)
        b_tur = calculate_b_tur(a, d)

        if b_tur is not None:
            a_values.append(a)
            b_hopf_values.append(b_hopf)
            b_tur_values.append(b_tur)

        a += h

    # Convert lists to numpy arrays for easier handling
    a_values = np.array(a_values)
    b_hopf_values = np.array(b_hopf_values)
    b_tur_values = np.array(b_tur_values)

    plt.figure(figsize=(8, 6))

    # Fill the regions with scientific colors
    plt.fill_between(a_values, b_hopf_values, b_tur_values, where=(b_hopf_values >= b_tur_values), color='lightyellow'
                     ,alpha=0.8)
    plt.fill_between(a_values, b_hopf_values, y2=np.max(b_hopf_values), color='lightgreen', alpha=0.8)  # Green
    plt.fill_between(a_values, -1, b_tur_values, where=(b_tur_values > 0), color='lightblue', alpha=0.5)  # Orange

    # Plot the curves
    plt.plot(a_values, b_hopf_values, label='Hopf Curve', color='green', lw=2.5, alpha=0.5)
    plt.plot(a_values, b_tur_values, label='Turing Curve', color='blue', lw=2.5, alpha=0.3)

    # Labels and appearance settings
    plt.xlabel('$a$', fontdict={'color': 'black', 'size': 30}, labelpad=5)
    plt.ylabel('$b$', fontdict={'color': 'black', 'size': 30}, rotation='horizontal', labelpad=23)
    plt.xticks(np.linspace(8, 16, num=5), fontsize=20)
    plt.yticks(np.linspace(0, 10, num=6), fontsize=20)
    plt.tick_params(axis='both', width=2, length=8)
    
    # Set plot limits
    plt.xlim(8, 16)
    plt.ylim(-1, 8)
    
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)

    # Mark a specific point
    plt.scatter(12, 1, color='black', s=150, alpha=1, zorder=5)

    # Legend and grid
    plt.legend(fontsize=24, facecolor='white',loc='upper left')
    plt.grid(False)
    plt.subplots_adjust(bottom=0.2,top=0.9,left=0.15)
    plt.savefig("bifurcation/Bifurcation_diagram.pdf")
    plt.show()

main()
