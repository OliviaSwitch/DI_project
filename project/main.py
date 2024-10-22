from ArtisticTransformer import ArtisticTransformer

import matplotlib.pyplot as plt

# Create transformer instance
transformer = ArtisticTransformer()

# Load an image
transformer.load_image('project/style_impressionniste.jpeg')

# # Try different styles
# impressionist = transformer.impressionist_style(brush_size=15)
# transformer.display_results(impressionist, "Impressionist Style")

# cubist = transformer.cubist_style(num_segments=100)
# transformer.display_results(cubist, "Cubist Style")

# pointillist = transformer.pointillist_style(dot_size=5, spacing=7)
# transformer.display_results(pointillist, "Pointillist Style")

# Generate all styles
print("Generating Impressionist style...")
impressionist = transformer.impressionist_style(brush_size=15)

print("Generating Cubist style...")
cubist = transformer.cubist_style(num_segments=100)

print("Generating Pointillist style...")
pointillist = transformer.pointillist_style(dot_size=5, spacing=7)

# Display all results
plt.figure(figsize=(20, 15))

# Original
plt.subplot(221)
plt.imshow(transformer.image)
plt.title("Original Image")
plt.axis('off')

# Impressionist
plt.subplot(222)
plt.imshow(impressionist)
plt.title("Impressionist Style")
plt.axis('off')

# Cubist
plt.subplot(223)
plt.imshow(cubist)
plt.title("Cubist Style")
plt.axis('off')

# Pointillist
plt.subplot(224)
plt.imshow(pointillist)
plt.title("Pointillist Style")
plt.axis('off')

plt.tight_layout()
plt.savefig('project/artistic_transformations.png')
plt.show()