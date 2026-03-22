// Very common open/barre shapes (simple starter set)
export const GUITAR_SHAPES = {
  "C":  { name: "C",  fingering: "x32010" },
  "Cm": { name: "Cm", fingering: "x35543" }, // barre
  "D":  { name: "D",  fingering: "xx0232" },
  "Dm": { name: "Dm", fingering: "xx0231" },
  "E":  { name: "E",  fingering: "022100" },
  "Em": { name: "Em", fingering: "022000" },
  "F":  { name: "F",  fingering: "133211" }, // barre
  "Fm": { name: "Fm", fingering: "133111" }, // barre
  "G":  { name: "G",  fingering: "320003" },
  "Gm": { name: "Gm", fingering: "355333" }, // barre
  "A":  { name: "A",  fingering: "x02220" },
  "Am": { name: "Am", fingering: "x02210" },
  "B":  { name: "B",  fingering: "x24442" }, // barre
  "Bm": { name: "Bm", fingering: "x24432" }, // barre
};

// fallback: just show chord name if not in map
export function getGuitarShape(chord) {
  return GUITAR_SHAPES[chord] || { name: chord, fingering: "------" };
}