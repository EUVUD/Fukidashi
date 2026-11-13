// One hue per region, shared by its box on the image, its row in the list and its mentions in the trace panel.
export const hue = (index: number) => `hsl(${(index * 67) % 360} 70% 48%)`
